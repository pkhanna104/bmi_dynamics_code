import analysis_config
from online_analysis import util_fcns, generate_models_utils, plot_generated_models

from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib as mpl

import copy, pickle
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats
from collections import defaultdict 
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.4, style='white')

###### notebook fcns #######
def extract_trans_stats(include_shuffle = True, only_potent = False):
    ''' 
    method to get accuracy of angle / magnitude
    '''
    all_stats_dict = {}
    save = False
    model_set_number = 7
    mod_type = 2 ### General dynamics 
    dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'

    for animal in ['jeev', 'grom']:
        if animal == 'grom':
            ndays = 9
        elif animal == 'jeev':
            ndays = 4
        
        stats_dict = defaultdict(list)
        pot_stats_dict = defaultdict(list)
            
        for day in range(ndays):

            if include_shuffle:
                R2_dA, R2_dM, perc_corr, R2_dA_shuff, R2_dM_shuff, perc_corr_shuff = plot_scatters(animal, model_set_number, dyn_model, save, 
                                                 day, minobs = 15, model_type = mod_type, only_potent = only_potent, include_shuffle = include_shuffle)
            else:
                R2_dA, R2_dM, perc_corr = plot_scatters(animal, model_set_number, dyn_model, save, 
                                                 day, minobs = 15, model_type = mod_type, only_potent = only_potent)
                        
            #### Save this stuff, skip across condition ####
            stats_dict['r2ang', 'trans'].append(R2_dA)
            stats_dict['r2mag', 'trans'].append(R2_dM)

            stats_dict['pc', 'mag', 'trans'].append(perc_corr['mag'])
            stats_dict['pc', 'ang_pw', 'trans'].append(perc_corr['ang_pairwise'])
        
            if include_shuffle:
                stats_dict['r2ang', 'trans_shuff'].append(R2_dA_shuff)
                stats_dict['r2mag', 'trans_shuff'].append(R2_dM_shuff)

                stats_dict['pc', 'mag', 'trans_shuff'].append(perc_corr_shuff['mag'])
                stats_dict['pc', 'ang_pw', 'trans_shuff'].append(perc_corr_shuff['ang_pairwise'])

        ##### Save this ######
        all_stats_dict[animal] = copy.deepcopy(stats_dict)

    return all_stats_dict

def plot_barplots_new_shuff(all_stats_dict):

    models_colors = [[100, 100, 100], [255, 0, 0]]
    models_colors = [np.array(m)/255. for m in models_colors]
    xlab = ['shuffled', 'general\ndynamics']
    for animal in all_stats_dict.keys():

        for pci in [None, 'pc']:
        
            #### Get R2 of model vs. shuffle for dAngle ####
            fR2, axR2 = plt.subplots(nrows = 2, figsize = (3, 8))

            if pci is None:
                keys = ['ang', 'mag']
            elif pci == 'pc':
                keys = ['ang_pw', 'mag']

            #### For each of these keys ######
            for ik, key in enumerate(keys):

                #### Values from all stats dict #######
                if pci is None:
                    vals = np.hstack((all_stats_dict[animal]['r2'+key, 'trans']))
                    vals_shuff = all_stats_dict[animal]['r2'+key, 'trans_shuff']
                elif pci == 'pc':
                    vals = process_pc(all_stats_dict[animal]['pc', key, 'trans'])
                    vals_shuff = process_pc(all_stats_dict[animal]['pc', key, 'trans_shuff'])                
                    vals = np.hstack((vals))

                util_fcns.draw_plot(0, np.hstack((vals_shuff)), models_colors[0], 'white', axR2[ik], width = 0.9)
                axR2[ik].bar(1, np.mean(vals), width=0.9, color=models_colors[1])
                axR2[ik].errorbar(1, np.mean(vals), np.std(vals)/np.sqrt(len(vals)), marker = '|', color='k')

                #### Plot each day: 
                ndays = len(vals)
                sig = []
                mn = []
                for i_d in range(ndays):
                    axR2[ik].plot([0, 1], [np.mean(vals_shuff[i_d]), vals[i_d]], '-', color='gray', linewidth=1.)
                    mn.append(np.mean(vals_shuff[i_d]))
                    ix_sig = np.nonzero(vals_shuff[i_d] >= vals[i_d])[0]
                    sig.append(float(len(ix_sig)) / len(vals_shuff[i_d]))
                
                if np.all(np.array(sig) < 0.05): 
                    axR2[ik].plot([0, 1], [1.1*np.max(vals), 1.1*np.max(vals)], 'k-')
                    axR2[ik].text(0.5, 1.15*np.max(vals), '***', ha='center')
                axR2[ik].set_xlim([-.5, 1.5])
                axR2[ik].set_ylim([np.min(mn), 1.2*np.max(vals)])
                
                axR2[ik].set_xticks([0, 1])
                axR2[ik].set_xticklabels(xlab, rotation=45)
                print('Animal %s, pc=%s, key=%s sig'%(animal, str(pci), key))
                print(sig)

            fR2.tight_layout()
            fR2.savefig(analysis_config.config['fig_dir']+'%s_PC_%s_fig5_trans_r2_ang_mag.svg'%(animal, str(pci)))

def process_pc(pc_dict):
    ndays = len(pc_dict)
    vals = []
    for i_d in range(ndays):
        vals_i = []
        for i_k in pc_dict[i_d].keys():
            vals_i.append(float(np.sum(pc_dict[i_d][i_k])) / float(len(pc_dict[i_d][i_k])))
        vals.append(vals_i)
    return vals

def plot_barplots(all_stats_dict, include_shuffle):

    ndays = dict(grom=9, jeev=4)
    blue = np.array([39, 169, 225])/255.

    plot_types = ['trans']
    if include_shuffle:
        plot_types.append('trans_shuff')
    
    ########## Bar plots of metrics ###########
    for i_a, animal in enumerate(['jeev', 'grom']):
        lme = dict(ang_pc = [], ang_pc_shuff = [], mag_pc = [], mag_shuff = [], day = [], ang_pw = [], 
            ang_pw_shuff = [])

        for i_d in range(ndays[animal]):
            
            ### Percent Correct CW / CCW ###
            tmp = all_stats_dict[animal][2, 'pc', 'trans'][i_d]
            pc = float(tmp[0]) / float(tmp[1])
            lme['ang_pc'].append(pc)

            tmp = all_stats_dict[animal][2, 'pc', 'trans_shuff'][i_d]
            pc = float(tmp[0]) / float(tmp[1])
            lme['ang_pc_shuff'].append(pc)

            ##### Angle pairwise relationships #####
            tmp = all_stats_dict[animal][2, 'pc', 'ang_pw', 'trans'][i_d]
            pc = float(tmp[0]) / float(tmp[1])
            lme['ang_pw'].append(pc)

            tmp = all_stats_dict[animal][2, 'pc', 'ang_pw', 'trans_shuff'][i_d]
            pc = float(tmp[0]) / float(tmp[1])
            lme['ang_pw_shuff'].append(pc)

            ### Magnitude change; ####
            tmp = all_stats_dict[animal][2, 'pc', 'mag', 'trans'][i_d]
            pc = float(tmp[0])/float(tmp[1])
            lme['mag_pc'].append(pc)
            
            tmp = all_stats_dict[animal][2, 'pc', 'mag', 'trans_shuff'][i_d]
            pc = float(tmp[0])/float(tmp[1])
            lme['mag_shuff'].append(pc)
            
            lme['day'].append(i_d)

        val_ang = np.hstack(( np.hstack(( lme['ang_pc'] )), np.hstack((lme['ang_pc_shuff'])) ))
        val_ang_pw = np.hstack(( np.hstack(( lme['ang_pw'] )), np.hstack((lme['ang_pw_shuff'])) ))
        
        grp_ang = np.hstack(( np.zeros((ndays[animal])) , np.zeros((ndays[animal])) + 1 ))
        day_ang = np.hstack(( np.hstack(( lme['day'] )), np.hstack(( lme['day'] )) ))
        
        pv_ang, slp = util_fcns.run_LME(day_ang, grp_ang, val_ang)
        print('Animal %s,, Angle CW/CCW predictions, pv = %.4f, slp = %.4f' %(animal, pv_ang, slp))
        
        pv_ang_pw, slp = util_fcns.run_LME(day_ang, grp_ang, val_ang_pw)
        print('Animal %s,, Angle CW/CCW predictions PAIRWISE, pv = %.4f, slp = %.4f' %(animal, pv_ang_pw, slp))

        val_mag = np.hstack(( np.hstack(( lme['mag_pc'] )), np.hstack((lme['mag_shuff'])) ))
        pv_mag, slp = util_fcns.run_LME(day_ang, grp_ang, val_mag)
        print('Animal %s,, MAG predictions, pv = %.4f, slp = %.4f' %(animal, pv_mag, slp))
        
        pv_ast_ang, pv_ast_mag, pv_ast_ang_pw = get_stars([pv_ang, pv_mag, pv_ang_pw])

        ######### Bar plot ##########
        alpha = .7

        f, ax = plt.subplots(ncols = 3, figsize = (15, 5))
        ix0 = np.nonzero(grp_ang == 0)[0]
        ix1 = np.nonzero(grp_ang == 1)[0]

        ax[0].bar(0, np.mean(val_ang[ix0]), width = 1., color = blue, alpha = alpha)
        ax[0].bar(1, np.mean(val_ang[ix1]), width = 1., color = 'gray', alpha = alpha)

        ax[1].bar(0, np.mean(val_mag[ix0]), width = 1., color = blue, alpha = alpha)
        ax[1].bar(1, np.mean(val_mag[ix1]), width = 1., color = 'gray', alpha = alpha)

        ax[2].bar(0, np.mean(val_ang_pw[ix0]), width = 1., color = blue, alpha = alpha)
        ax[2].bar(1, np.mean(val_ang_pw[ix1]), width = 1., color = 'gray', alpha = alpha)

        ax[0].plot([0, 1], [.95, .95], 'k-')
        ax[0].text(.5, .96, pv_ast_ang)

        ax[1].plot([0, 1], [.95, .95], 'k-')
        ax[1].text(.5, .96, pv_ast_mag)

        ax[2].plot([0, 1], [.95, .95], 'k-')
        ax[2].text(.5, .96, pv_ast_ang_pw)

        for i_d in range(ndays[animal]):
            ix = np.nonzero(day_ang == i_d)[0]
            val_day = val_ang[ix]
            grp_day = grp_ang[ix]

            tmp = [val_day[grp_day == 0], val_day[grp_day==1]]
            ax[0].plot([0, 1], tmp, '-', color='gray')

            val_mday = val_mag[ix]
            tmp = [val_mday[grp_day == 0], val_mday[grp_day==1]]
            ax[1].plot([0, 1], tmp, '-', color='gray')

            val_angpw_day = val_ang_pw[ix]
            tmp = [val_angpw_day[grp_day == 0], val_angpw_day[grp_day==1]]
            ax[2].plot([0, 1], tmp, '-', color='gray')

        ax[0].set_ylabel('Percent Correct in CW/CCW direction\n vs. Shuffle')
        ax[1].set_ylabel('Percent Correct Mag prediction \n vs. Shuffle')
        ax[2].set_ylabel('Percent Correct Ang prediction \n vs. Shuffle')
        
        ax[0].set_ylim([0.4, 1.1])
        ax[1].set_ylim([0.4, 1.1])
        ax[2].set_ylim([0.4, 1.1])
        
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(['$y_t | y_{t-1}$', 'shuffle'], rotation = 45, fontsize=14)
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(['$y_t | y_{t-1}$', 'shuffle'], rotation = 45, fontsize=14)
        ax[2].set_xticks([0, 1])
        ax[2].set_xticklabels(['$y_t | y_{t-1}$', 'shuffle'], rotation = 45, fontsize=14)
        
        ax[0].set_title("Animal %s" %(animal))
        f.tight_layout()
        f.savefig(analysis_config.config['fig_dir3']+'%s_next_step_pred_real_vs_shuffle.eps' %(animal))
        # ax.set_ylabel('Perc Correct in CW/CCW direction vs. Chance')
        # ax.set_xlabel('Days')
        # ax.set_title("Animal %s, Gen Dyn, XTrans" %(animal))      

def _notebook_dump():

    def test_arc():
        import matplotlib.patches as mpatches
        f, ax = plt.subplots()

        center = (0., 0.)         # center of the circle
        radius = 3.
        theta1 = -125.
        theta2 = 125.
        width = 2.

        patch = mpatches.Wedge(center=center,
                               r=radius,
                               theta1=theta1,
                               theta2=theta2,
                               width=width,
                               # COMMON OPTIONS:
        #                        alpha=alpha,
                                fill=False,)
                                #facecolor='b')
        #                        hatch=fill_pattern,
        #                        ec=line_color,
        #                        lw=line_width,
        #                        joinstyle=join_style,
        #                        linestyle=line_style)

        ax.add_patch(patch)
        ax.set_title("matplotlib.patches.Wedge\n(width=None)", fontsize=14)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.plot([0, 3], [0, 0], 'k-')
        ax.plot([0, 0], [0, 1], 'k-')


    # ###### Bar Plots on stats #######
    # ### Gen vs. Condition specific plots together, 
    # ### Trans / xcond plots together; 
    
    # xlab = []    
    # ### L = Angle, R = Mag; 
    # ### Rows = [slp, rv, r2]
    # ndays = dict(grom=9, jeev=4)

    # for i_a, animal in enumerate(['grom', 'jeev']):
    #     stats_dict = all_stats_dict[animal]
    #     colors = dict(trans='purple', xcond='darkblue')
        
    #     f, ax = plt.subplots(ncols = 2, nrows = 3, figsize = (8, 10))

    #     ###  Metric x ang/mag x days
    #     day_lines = np.zeros((3, 2, ndays[animal]))

    #     for i_mt, mt in enumerate([2]):
    #         for i_am, am in enumerate(['ang', 'mag']):
    #             for j, jnm in enumerate(['trans']): ### Focus on transitions 
    #                 for i_s, stats in enumerate([stats_dict]): ### Focus on stats 

    #                     ### Get the temp stat ###
    #                     tmp = np.vstack((stats[mt, am, jnm]))

    #                     for i_m, met in enumerate(['Slp', 'R-line', 'R2-err']):
    #                         axi = ax[i_m, i_am]

    #                         if i_s == 0:
    #                             axi.bar(3*j + i_mt + .5*i_s, np.mean(tmp[:, i_m]), color = colors[jnm], 
    #                                 alpha = (1. - 0.5*i_mt), width = .5)
    #                         else:
    #                             axi.bar(3*j + i_mt + .5*i_s, np.mean(tmp[:, i_m]), color = 'gray', 
    #                                 alpha = (1. - 0.5*i_mt), width = .5)

    #                         axi.errorbar(3*j + i_mt + .5*i_s, np.mean(tmp[:, i_m]), 
    #                                      np.std(tmp[:, i_m])/np.sqrt(len(tmp)),
    #                                      marker = '|', color = 'k')

    #                         axi.set_ylabel(met)
    #                         axi.set_title(am)
    #                         day_lines[i_m, i_am, :] = tmp[:, i_m]
    #                         xlab.append('Pred. %s\nMod Type %d'%(jnm, mt))

    #     f.tight_layout()
    pass

def get_stars(pv_list):
    tmp = []
    for pv in pv_list:
        if pv < .0001:
            tmp.append('***')
        elif pv < .01:
            tmp.append('**')
        elif pv < .05:
            tmp.append('*')
        else:
            tmp.append('n.s., pv=%.3f' %(pv))
    return tmp

def plot_scatters(animal, model_set_number, dyn_model, save, day, minobs = 15, minobs2 = 0,
                 model_type = 2, only_potent = False, include_shuffle = False):
    
    '''
    get data to make scatters of predicted vs. true angle 
    '''
    
    data = preproc(animal, model_set_number, dyn_model, day, model_type = model_type, 
                                minobs = minobs, minobs2 = minobs2, only_potent = only_potent, 
                                include_shuffle = include_shuffle)
    data['min_obs'] = minobs
    data['save'] = save
    data['save_dir'] = save_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/pred_scatters/'
    data['prefix'] = 'xtrans_'+data['prefix']

    R2_dA, R2_dM, perc_corr = plot_real_vs_pred_dAngle_dMag(**data)
    #ang_stats, mag_stats, perc_corr, fax = plot_real_vs_pred_dAngle_dMag(**data)

    if include_shuffle:
        data_shuf = copy.deepcopy(data)
        
        ### Move the pred_spks_shuff to the pred_spks
        data_shuf['pred_spks'] = data_shuf['pred_spks_shuff']
        data_shuf['pred_push'] = data_shuf['pred_push_shuff']
        data_shuf['shuff_data'] = True

        R2_dA_shuff, R2_dM_shuff, perc_corr_shuff = plot_real_vs_pred_dAngle_dMag(**data_shuf)
        # ax = fax_shuf[1]
        # if type(ax) is np.ndarray:
        #     for axi in ax:
        #         tmp = axi.get_title()
        #         axi.set_title(tmp+'\n Model type'+str(model_type)+' SHUFF, only pot = '+str(only_potent)+'\n Day = '+str(day))
        # else:
        #     tmp = ax.get_title()
        #     ax.set_title(tmp+'\n Model type'+str(model_type)+' SHUFF, only pot = '+str(only_potent)+'\n Day = '+str(day))


    # ax = fax[1]
    # if type(ax) is np.ndarray:
    #     for axi in ax:
    #         tmp = axi.get_title()
    #         axi.set_title(tmp+'\n Model type'+str(model_type)+', only pot = '+str(only_potent)+'\n Day = '+str(day))
    # else:
    #     tmp = ax.get_title()
    #     ax.set_title(tmp+'\n Model type'+str(model_type)+', only pot = '+str(only_potent)+'\n Day = '+str(day))
    
    # if include_shuffle:
    #     return ang_stats, mag_stats, perc_corr, ang_shuff, mag_shuff, perc_corr_shuff
    # else:
    #     return ang_stats, mag_stats, perc_corr
    if include_shuffle:
        return R2_dA, R2_dM, perc_corr, R2_dA_shuff, R2_dM_shuff, perc_corr_shuff
    else:
        return R2_dA, R2_dM, perc_corr

def plot_scatters_xcond(animal, model_set_number, dyn_model, save, day, minobs = 15, minobs2 = 0,
                       model_type = 2,  only_potent = False):
    
    data = preproc(animal, model_set_number, dyn_model, day, model_type = model_type, 
                                minobs = minobs, minobs2 = minobs2, only_potent = only_potent)
    data['min_obs'] = minobs
    data['save'] = save
    data['save_dir'] = save_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/pred_scatters/'
    data['prefix'] = 'xcond_'+data['prefix']
    ang_stats, mag_stats = plot_dAngle_dMag_xcond(**data)
    return ang_stats, mag_stats

def make_tg_compare_plots(animal, model_set_number, save, scale_rad = 2., 
                   dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0', day = 0, minobs = 10, minobs2 = 3,
                   model_type = 2.,
                   save_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/action_distributions_w_preds_pls_angmagsubs'):
    
    data = preproc(animal, model_set_number, dyn_model, day, model_type = model_type, 
                                minobs = minobs, minobs2 = minobs2)
    
    data['arrow_scale'] = .01; 
    data['lims'] = 8
    data['save'] = save
    data['save_dir'] = save_dir 
    data['add_ang_mag_subplots'] = True; 
    data['scale_rad'] = scale_rad; 
    
    # ### Go through adjacent targets 
    for tg0 in range(8):
        tg1 = np.mod(tg0+1, 8)
        
        data['targ0'] = tg0; 
        data['targ1'] = tg1; 
        plot_pred(**data)

def make_ALL_plots(animal, model_set_number, save, scale_rad = 2., 
                   dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0', day = 0, minobs = 10,
                   model_type = 2.,
                   save_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/data/action_distributions_all_cond_w_preds_pls_angmagsubs'):
    
    data = preproc(animal, model_set_number, dyn_model, day, model_type = model_type, 
                                minobs = 0, minobs2 = minobs)
    data['arrow_scale'] = 0.01
    data['lims'] = 8; 
    data['save'] = save
    data['save_dir'] = save_dir; 
    data['scale_rad'] = scale_rad; 
    data['add_ang_mag_subplots'] = True
    plot_pred_across_full_day(**data)

########### Plot nice example #####
#dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(6), 'rb'))
def plot_example_dAng_dMag(animal, day, angle_og, mag_og, minobs = 10, dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0', dat=None):
    
    data = preproc(animal, 6, dyn_model, day, model_type = 2, 
                                minobs = 0, minobs2 = minobs, dat=dat)

    T = len(data['command_bins'])
    ix_command = np.nonzero(np.logical_and(data['command_bins'][:, 0] == mag_og, data['command_bins'][:, 1] == angle_og))[0]
    ix_com_tp1 = ix_command + 1; 

    ix_keep = np.nonzero(ix_com_tp1 < T)[0]
    ix_command = ix_command[ix_keep]
    ix_com_tp1 = ix_com_tp1[ix_keep]

    ix_keep2 = np.nonzero(data['bin_num'][ix_com_tp1] > np.min(data['bin_num']))[0]
    ix_command = ix_command[ix_keep2]
    ix_com_tp1 = ix_com_tp1[ix_keep2]

    assert(np.all(data['command_bins'][ix_command, 0] == mag_og))
    assert(np.all(data['command_bins'][ix_command, 1] == angle_og))
    assert(np.all(data['bin_num'][ix_com_tp1] > np.min(data['bin_num'])))
    assert(np.all(ix_command == ix_com_tp1 - 1))

    ######## Plot 3 circles ##########
    true_trans = dict()
    pred_trans = dict()
    command_mn = []
    added_tuples = []

    for _, (ix, ix_og) in enumerate(zip(ix_com_tp1, ix_command)):
        com = data['command_bins'][ix, :]
        if tuple(com) in added_tuples:
            pass
        else:
            true_trans[tuple(com)] = []
            pred_trans[tuple(com)] = []
            added_tuples.append(tuple(com))
        true_trans[tuple(com)].append(data['push'][ix, :])
        pred_trans[tuple(com)].append(data['pred_push'][ix, :])
        command_mn.append(data['push'][ix_og, :])

    ######## Segment mean ##########
    segment_mn = np.mean(np.vstack((command_mn)), axis=0)

    dA_dM = dict(dA = [], dM = [], trans = [])
    dA_dM_true = dict(dA = [], dM = [], trans = [])
    cont = False
    for it, trans in enumerate(added_tuples):
        tmp = np.vstack((true_trans[trans]))
        if len(tmp) > minobs:
            true_ = np.mean(np.vstack((true_trans[trans])), axis=0)
            pred_ = np.mean(np.vstack((pred_trans[trans])), axis=0)
        
            cont = True
            dA, dM = get_diffs(true_, segment_mn)
            cp = np.sign(np.cross(segment_mn, true_))
            dA = cp*dA

            pred_dA, pred_dM = get_diffs(pred_, segment_mn)
            pred_cp = np.sign(np.cross(segment_mn, pred_))
            pred_dA = pred_cp*np.abs(pred_dA)

            ##### Append to end ###########
            dA_dM['dA'].append(pred_dA)
            dA_dM['dM'].append(pred_dM)
            dA_dM['trans'].append(trans)

            dA_dM_true['dA'].append(dA)
            dA_dM_true['dM'].append(dM)
            dA_dM_true['trans'].append(trans)

    if cont:
        ####### Plots ######### 
        f, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
        t = np.linspace(0, 2*np.pi, 1000)
        angles1 = np.linspace(0., 2*np.pi, 9.) - np.pi/8
        angles2 = np.linspace(0., 2*np.pi, 9.) + np.pi/8
        angles1 = angles1[:-1]
        angles2 = angles2[:-1]

        mag1 = np.hstack(([0], data['mag_bound']))
        dm = data['mag_bound'][-1] - data['mag_bound'][-2]
        mag2 = np.hstack((data['mag_bound'], data['mag_bound'][-1] + dm))
        
        
        ###### Draw circles / lines #######
        for axi in ax.reshape(-1):
            for magi in mag2:
                ### Plot circles + angles; 
                axi.plot(magi*np.cos(t), magi*np.sin(t), 'k-', linewidth=1.)

            ##### Angle plots ######
            for ang in angles1:
                axi.plot([0, magi*np.cos(ang)], [0, magi*np.sin(ang)], 'k-', linewidth=1.)
            axi.axis('square')

        angles1 = angles1 / np.pi * 180.
        angles2 = angles2 / np.pi * 180.
        
        ##### Make colormaps #######
        labs = [['True dAngle', 'True dMag'], ['Pred dAngle', 'Pred dMag']]
        for rowi, datrow in enumerate([dA_dM_true, dA_dM]):

            for coli, key in enumerate(['dA', 'dM']):

                vals = datrow[key]
                entries = datrow['trans']

                cmap = plt.get_cmap('viridis')
                cNorm = colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)

                for iv, (vl, ent) in enumerate(zip(vals, entries)):
                    
                    mag, ang = ent; 
                    colorVal = scalarMap.to_rgba(vl)
                    patch = mpatches.Wedge(center=(0., 0.),
                                           r=mag2[mag],
                                           theta1=angles1[ang],
                                           theta2=angles2[ang],
                                           width=mag2[mag]-mag1[mag],
                                           fill=True, facecolor=colorVal)
                    ax[coli, rowi].add_patch(patch)

                scalarMap.set_array([])
                cbar = f.colorbar(scalarMap, ax = ax[coli, rowi], orientation='horizontal')
                cbar.ax.set_title(labs[rowi][coli])
        
                #### Add one last wedge #####
                patch = mpatches.Wedge(center=(0., 0.),
                                       r=mag2[mag_og],
                                       theta1=angles1[angle_og],
                                       theta2=angles2[angle_og],
                                       width=mag2[mag_og]-mag1[mag_og],
                                       fill=False, lw= 3., ec='k')
                ax[coli, rowi].add_patch(patch)

        f.tight_layout()
        f.savefig(analysis_config.config['fig_dir']+'%s_day%d_mag%d_ang%d_fig5_eg_disc.svg'%(animal, day, mag_og, angle_og))

def plot_disc_colorless():
    ####### Plots ######### 
    f, ax = plt.subplots(figsize=(4, 4))
    t = np.linspace(0, 2*np.pi, 1000)
    angles1 = np.linspace(0., 2*np.pi, 9.) - np.pi/8
    angles1 = angles1[:-1]
    mag2 = [1, 2, 3, 4]
    
    ###### Draw circles / lines #######
    for magi in mag2:
        ### Plot circles + angles; 
        ax.plot(magi*np.cos(t), magi*np.sin(t), 'k-', linewidth=1.)

    ##### Angle plots ######
    for ang in angles1:
        ax.plot([0, magi*np.cos(ang)], [0, magi*np.sin(ang)], 'k-', linewidth=1.)
        ax.axis('square')
    f.savefig(analysis_config.config['fig_dir']+'disc.svg')

########### Plot action plots #######
def plot_pred(command_bins=None, push=None, trg=None, tsk=None, bin_num=None, mag_bound=None, targ0 = 0., targ1 = 1., min_obs = 20, min_obs2 = 3,
             arrow_scale = .004, lims = 5, prefix = '', save = False, save_dir = None, pred_push = None, scale_rad = 2.,
             add_ang_mag_subplots = False, **kwargs): 
    
    ### Get the mean angle / magnitudes ###
    mag_bound = scale_rad*np.hstack((mag_bound))
    ang_mns = np.linspace(0, 2*np.pi, 9)
    mag_mns = [np.mean([0., mag_bound[0]])]
    for i in range(2):
        mag_mns.append(np.mean([mag_bound[i], mag_bound[i+1]]))
    mag_mns.append(np.mean([mag_bound[2], mag_bound[2] + 2]))
    
    ### INDEX LEVEL 1 -- Get all the trg0 / trg1 data; 
    ix0 = np.nonzero(np.logical_and(trg == targ0, tsk == 0))[0]
    ix1 = np.nonzero(np.logical_and(trg == targ1, tsk == 1))[0]
    
    ### For the CO task go through commands and get the ones with enough observations ;
    com0 = command_bins[ix0, :]
    com1 = command_bins[ix1, :]
    
    psh0 = push[ix0, :]
    psh1 = push[ix1, :]
    
    if pred_push is None:
        pred_psh0 = None
        pred_psh1 = None
    else:
        pred_psh0 = pred_push[ix0, :]
        pred_psh1 = pred_push[ix1, :]
    
    bn0 = bin_num[ix0]
    bn1 = bin_num[ix1]
    
    trg0 = trg[ix0]
    trg1 = trg[ix1]
    
    ### Go through -- which bins have enough min_obs in both tasks; 
    keep = []
    for m_ in range(4):
        for a_ in range(8):
            
            ### INDEX LEVEL 2 -- Indices to get the mag / angle from each task 
            i0 = np.nonzero(np.logical_and(com0[:, 0] == m_, com0[:, 1] == a_))[0]
            i1 = np.nonzero(np.logical_and(com1[:, 0] == m_, com1[:, 1] == a_))[0]
            
            ### Save these ###
            if np.logical_and(len(i0) >= min_obs, len(i1) >= min_obs):
                
                ### Add the m / a / indices for this so we can go through them later; 
                keep.append([m_, a_, i0.copy(), i1.copy()])
    
    ### Now for each for these, plot the main thing; 
    for i_m, (mi, ai, i0, i1) in enumerate(keep):           
        
        ### Get mean neural push for each task ###
        segment_mn0 = np.mean(psh0[i0, :], axis=0)
        segment_mn1 = np.mean(psh1[i1, :], axis=0)

        ### ADJUSTMENT TO LEVEL 2 -- Now get the NEXT action commands ###
        i0p1 = i0 + 1; 
        i1p1 = i1 + 1; 
        
        ### If exceeds length after adding "1"
        if i0p1[-1] > (len(bn0) -1):
            i0p1 = i0p1[:-1]
        if i1p1[-1] > (len(bn1) -1):
            i1p1 = i1p1[:-1]
        ### Remove the commands that are equal to the minimum bin_cnt (=1) 
        ### because they spilled over from the last trial ###
        kp0 = np.nonzero(bn0[i0p1] > np.min(bin_num))[0]
        kp1 = np.nonzero(bn1[i1p1] > np.min(bin_num))[0]
        
        print('Min Bin num %d' %(np.min(bin_num)))
        print('Len rm %d, len pre rm %d' %(len(i0p1), len(kp0)))

        ### Keep these guys ###
        i0p1 = i0p1[kp0]
        i1p1 = i1p1[kp1]
        
        ### Arrows dict 
        ### We need to create this because we're not sure how big to maek the colormap when plotting; 
        arrows_dict = {}
        arrows_dict[0] = []
        arrows_dict[1] = []
        
        pred_arrows_dict = {}
        pred_arrows_dict[0] = []
        pred_arrows_dict[1] = []
        
        index_dict = {}
        index_dict[0] = []
        index_dict[1] = []

        dA_dict = {}
        dA_dict[0] = []
        dA_dict[1] = []

        dM_dict = {}
        dM_dict[0] = []
        dM_dict[1] = []
        
        ### These should all be the same target still ###
        assert(np.all(trg0[i0p1] == targ0))
        assert(np.all(trg1[i1p1] == targ1))
        
        ### Iterate through the tasks ###

        for mp in range(4):
            for ap in range(8):

                ### LEVEL 3 -- OF THE NEXT ACTION COMMANDS, which match the action #####
                j0 = np.nonzero(np.logical_and(com0[i0p1, 0] == mp, com0[i0p1, 1] == ap))[0]
                j1 = np.nonzero(np.logical_and(com1[i1p1, 0] == mp, com1[i1p1, 1] == ap))[0]

                for i_, (index, index_og, pshi, predpshi, segmn) in enumerate(zip([j0, j1], [i0p1, i1p1], [psh0, psh1],
                                                                [pred_psh0, pred_psh1], [segment_mn0, segment_mn1])):
                    if len(index) >= min_obs2:
                        print('Adding followup tsk %d, m %d, a %d' %(i_, mp, ap))

                        ### Great plot this;
                        mn_next_action = np.mean(pshi[index_og[index], :], axis=0)
                        xi = mag_mns[mp]*np.cos(ang_mns[ap])
                        yi = mag_mns[mp]*np.sin(ang_mns[ap])
                        vx = mn_next_action[0];
                        vy = mn_next_action[1]; 

                        arrows_dict[i_].append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(vx), copy.deepcopy(vy), len(index)])
                        index_dict[i_].append(len(index))
                        
                        #### Predicted Action ####
                        if predpshi is None:
                            pass
                        else:
                            pred_mn_next_action = np.mean(predpshi[index_og[index], :], axis=0)
                            
                            #### Get the error associated with the predicted action ####
                            dA, dM = get_diffs(pred_mn_next_action, segmn)
                            dA_dict[i_].append(dA)
                            dM_dict[i_].append(dM)

                            p_vx = pred_mn_next_action[0]
                            p_vy = pred_mn_next_action[1]
                            
                            pred_arrows_dict[i_].append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(p_vx), copy.deepcopy(p_vy), dA, dM, mp, ap])
        
        #import pdb; pdb.set_trace()
        ### Now figure out the color lists for CO / OBS separately; 
        tmpx = 0; 
        if len(index_dict[0]) > 0:
            mx_co = np.max(np.hstack((index_dict[0])))
            mn_co = np.min(np.hstack((index_dict[0])))
            co_cols = np.linspace(0., 1., mx_co - mn_co + 1)
            co_colors = [cm.viridis(x) for x in co_cols]
            print('Co mx = %d, mn = %d' %(mx_co, mn_co))
            tmpx += 1
        
        if len(index_dict[1]) > 0:
            mx_ob = np.max(np.hstack((index_dict[1])))
            mn_ob = np.min(np.hstack((index_dict[1])))
            obs_cols = np.linspace(0., 1., mx_ob - mn_ob+1)
            obs_colors = [cm.viridis(x) for x in obs_cols]
            print('Obs mx = %d, mn = %d' %(mx_ob, mn_ob))
            tmpx += 1

        if tmpx == 2:

            ######## Make a figure centered on this command; ###########
            if add_ang_mag_subplots:
                fig, ax_all = plt.subplots(ncols = 2, nrows = 3, figsize = (10, 150/8.))
            else:
                fig, ax_all = plt.subplots(ncols = 2, figsize = (10, 50/8.))
                ax_all = ax_all[np.newaxis, :]
                fig.subplots_adjust(bottom=0.3)
            
            for axi in ax_all.reshape(-1):
                axi.axis('equal')

            for ia, (ax, tgi) in enumerate(zip(ax_all[0, :], [targ0, targ1])):
                
                ### Set the axis to square so the arrows show up correctly 
                ax.axis('equal')
                title_string = 'cotg%d_obstg%d_mag%d_ang%d' %(targ0, targ1, mi, ai)

                ### Plot the division lines; 
                for A in np.linspace(-np.pi/8., (2*np.pi) - np.pi/8., 9):
                    for axi in ax_all[:, ia]:
                        axi.plot([0, lims*np.cos(A)], [0, lims*np.sin(A)], 'k-', linewidth=.5)
                
                ### Plot the circles 
                t = np.arange(0, np.pi * 2.0, 0.01)
                tmp = np.hstack(([0], mag_bound, [mag_bound[2] + 2]))
                for mm in tmp:
                    x = mm * np.cos(t)
                    y = mm * np.sin(t)
                    for axi in ax_all[:, ia]:
                        axi.plot(x, y, 'k-', linewidth = .5)

            ### Plot this mean neural push as big black arrow ###
            ax_all[0, 0].quiver(mag_mns[mi]*np.cos(ang_mns[ai]), mag_mns[mi]*np.sin(ang_mns[ai]), segment_mn0[0], segment_mn0[1], 
                      width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')
            
            ax_all[0, 1].quiver(mag_mns[mi]*np.cos(ang_mns[ai]), mag_mns[mi]*np.sin(ang_mns[ai]), segment_mn1[0], segment_mn1[1], 
                      width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')  



            #### Get a unified colorbar; 
            ##### Get color for angle / magnitude here; #######
            if add_ang_mag_subplots:
                tmp_a = np.hstack(( np.hstack(( dA_dict[0] )), np.hstack(( dA_dict[1] )) ))
                tmp_m = np.hstack(( np.hstack(( dM_dict[0] )), np.hstack(( dM_dict[1] )) ))
                npts = 10.
                tmp_cols = np.linspace(0., 1., npts)
                sub_colors = [cm.viridis(x) for x in tmp_cols]
                
                ### These are the center colors for each of the segments; 
                ang_dict = np.linspace(np.min(tmp_a), np.max(tmp_a), npts)
                mag_dict = np.linspace(np.min(tmp_m), np.max(tmp_m), npts)


            #### Now plot the ind. plots ###
            for tsk, (mnN, cols, segmn) in enumerate(zip([mn_co, mn_ob], [co_colors, obs_colors], [segment_mn0, segment_mn1])):
                
                cmap = mpl.cm.viridis
                num_corr = [0, 0]

                print('Len arrows_dict[%d]: %d' %(tsk, len(arrows_dict[tsk])))
                ### go through all the arrows;
                for arrow in arrows_dict[tsk]:
                    
                    ### Parse the parts 
                    xi, yi, vx, vy, N = arrow; 
                    #import pdb; pdb.set_trace()
                    ### Plot it;
                    ax_all[0, tsk].quiver(xi, yi, vx, vy,
                                      width = arrow_scale, color = cols[N - mnN], 
                                        angles='xy', scale=1, scale_units='xy')
                
                for pred_arrow in pred_arrows_dict[tsk]:
                    
                    ### Parse the parts
                    pxi, pyi, pvx, pvy, da, dm, mp, ap = pred_arrow
                    
                    #### Is this correct direction ? ######
                    corr_dir = assess_corr_dir([mi, ai], [mp, ap], segmn, np.array([pvx, pvy]))

                    if corr_dir is None: 
                        ax_all[0, tsk].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='gray', facecolor='none', 
                                          linewidth = 1.)

                    elif corr_dir == True: 
                        ### Plot it; 
                        num_corr[0] += 1
                        num_corr[1] += 1
                        
                        ax_all[0, tsk].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='r', facecolor='none', 
                                          linewidth = 1.)


                    elif corr_dir == False: 
                        num_corr[1] += 1
                        ### Plot it; 
                        ax_all[0, tsk].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          facecolor='r', alpha = 0.4, linewidth=0.)


                    #### Plot the predictions; 
                    if add_ang_mag_subplots:
                        x_pts, y_pts = get_pts_in_sector(mp, ap, mag_bound)

                        ### Which color ix is da and dr; 
                        ang_col_ix = np.argmin(np.abs(ang_dict - da))
                        mag_col_ix = np.argmin(np.abs(mag_dict - dm))

                        ax_all[1, tsk].plot(x_pts, y_pts, '.', color = sub_colors[ang_col_ix])
                        ax_all[2, tsk].plot(x_pts, y_pts, '.', color = sub_colors[mag_col_ix])

                if num_corr[1] > 0:
                    pc = int(100*(float(num_corr[0])/float(num_corr[1])))
                else:
                    pc = -1

                if tsk == 0:
                    ax_all[0, tsk].set_title('Task %d, Targ %d Mag %d, Ang %d, PC = %d' %(tsk, targ0, mi, ai, pc))
                elif tsk == 1:
                    ax_all[0, tsk].set_title('Task %d, Targ %d Mag %d, Ang %d, PC = %d' %(tsk, targ1, mi, ai, pc))
                
                ##### Now plot the colorbars for Angle / Magnitude; 
                if add_ang_mag_subplots:
                    for _, (r, dicts, clab) in enumerate(zip([1, 2], [ang_dict, mag_dict], ['dAngle (deg)', 'dMag'])):
                        
                        rfrombott = 2 - r
                        cax1 = fig.add_axes([tsk*.45 + .15, (rfrombott*.27) + .1, 0.3, 0.01])

                        if r == 1:
                            norm = mpl.colors.Normalize(vmin=dicts[0]/np.pi*180, vmax=dicts[-1]/np.pi*180)
                            tmp = np.array([dicts[0], np.mean(dicts), dicts[-1]])/np.pi*180
                            dt = 1/20.*(tmp[-1] - tmp[0])
                            tmp[0] += dt
                            tmp[-1] -= dt
                            tmp = tmp.astype(int)

                        else:
                            norm = mpl.colors.Normalize(vmin=dicts[0], vmax=dicts[-1])
                            tmp = np.array([dicts[0], np.mean(dicts), dicts[-1]])
                            dt = 1/20.*(dicts[-1] - dicts[0])
                            tmp[0] += dt 
                            tmp[-1] -= dt
                            tmp = 1/100.*(np.round(100*tmp).astype(int))

                        cb2 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap,
                                            norm=norm, orientation='horizontal')
                        cb2.set_ticks(tmp)
                        cb2.set_ticklabels(tmp)
                        cb2.set_label(clab)


            for ia, ax in enumerate(ax_all.reshape(-1)):
                ax.set_xlim([-lims, lims])
                ax.set_ylim([-lims, lims])
                ax.set_xticks([])
                ax.set_xticklabels([])

            ### Set colorbar for the main distriubitons 
            if mn_co == mx_co:
                norm = mpl.colors.Normalize(vmin=mn_co, vmax=mx_co+.1)
            else:
                norm = mpl.colors.Normalize(vmin=mn_co, vmax=mx_co)
            if add_ang_mag_subplots:
                cax0 = fig.add_axes([0*.45 + .15, (2*.27) + .1, 0.3, 0.01])
            else:
                cax0 = fig.add_axes([0.1, 0.1, 0.3, 0.05])
            cb1 = mpl.colorbar.ColorbarBase(cax0, cmap=cmap,
                                norm=norm, orientation='horizontal',
                                boundaries = np.arange(mn_co-0.5, mx_co+1.5, 1.))
            cb1.set_ticks(np.arange(mn_co, mx_co+1))
            cb1.set_ticklabels(np.arange(mn_co, mx_co+1))
            cb1.set_label('Counts')

            if mn_ob == mn_ob:
                norm = mpl.colors.Normalize(vmin=mn_ob, vmax=mx_ob+.1)
            else:
                norm = mpl.colors.Normalize(vmin=mn_ob, vmax=mx_ob)
            if add_ang_mag_subplots:
                cax1 = fig.add_axes([1*.45 + .15, (2*.27) + .1, 0.3, 0.01])
            else:
                cax1 = fig.add_axes([0.6, .1, .3, .05])
            cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap,
                                norm=norm, orientation='horizontal',
                                boundaries = np.arange(mn_ob-0.5, mx_ob+1.5, 1.))
            cb1.set_label('Counts')
            cb1.set_ticks(np.arange(mn_ob, mx_ob+1))
            cb1.set_ticklabels(np.arange(mn_ob, mx_ob+1)) 
        
            ### Save this ? 
            if save:
                fig.savefig(save_dir+'/'+prefix+'_'+title_string+'.png')

def plot_pred_across_full_day(command_bins=None, push=None, bin_num=None, mag_bound=None, min_obs2 = 20,
             arrow_scale = .004, lims = 5, prefix = '', save = False, save_dir = None, 
             pred_push = None, scale_rad = 2., add_ang_mag_subplots = False, **kwargs): 
    '''
    command_bins -- discretized commands over all bins 
    push -- continuous pushses
    bin_num -- indices within the trial 
    mag_bound -- magnitude boundaries
    min_obs2 -- number of counts of future actions needed to plot; 

    pred_push -- general dynamics predictions; 
    scale_rad -- how to scale out the radius for better visualization; 
    add_ang_mag_subplots -- add 2 additional subplots that visualize dAngle, dMag 
        as dots; 
    '''

    ### Get the mean angle / magnitudes ###
    mag_bound = scale_rad*np.hstack((mag_bound))
    ang_mns = np.linspace(0, 2*np.pi, 9)
    mag_mns = [np.mean([0., mag_bound[0]])]
    for i in range(2):
        mag_mns.append(np.mean([mag_bound[i], mag_bound[i+1]]))
    mag_mns.append(np.mean([mag_bound[2], mag_bound[2] + 2]))
    
    ### Go through -- which bins have enough min_obs in both tasks; 
    keep = []
    for m in range(4):
        for a in range(8):
            
            ### INDEX LEVEL 2 -- Indices to get the mag / angle from each task 
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]
            
            ### Add the m / a / indices for this so we can go through them later; 
            keep.append([m, a, ix.copy()])
    
    ### Now for each for these, plot the main thing; 
    for i_m, (m, a, ixi) in enumerate(keep):
        
        ### Make a figure centered on this command; 
        if add_ang_mag_subplots:
            fig, ax_all = plt.subplots(ncols = 3, figsize = (18, 6))
            fig.subplots_adjust(bottom=0.1)
        
        else:
            fig, ax_all = plt.subplots(figsize = (8, 10))
            fig.subplots_adjust(bottom=0.3)
            ax_all = [ax_all]
        
        ### Set the axis to square so the arrows show up correctly
        for ax in ax_all:
            ax.axis('equal')

        ### Used to save later; 
        title_string = 'ALL_mag%d_ang%d' %(m, a)

        ### Plot the division lines; 
        for A in np.linspace(-np.pi/8., (2*np.pi) - np.pi/8., 9):
            for ax in ax_all:
                ax.plot([0, lims*np.cos(A)], [0, lims*np.sin(A)], 'k-', linewidth=.5)
            
        ### Plot the circles 
        t = np.arange(0, np.pi * 2.0, 0.01)
        tmp = np.hstack(([0], mag_bound, [mag_bound[2] + 2]))
        for mm in tmp:
            x = mm * np.cos(t)
            y = mm * np.sin(t)
            for ax in ax_all:
                ax.plot(x, y, 'k-', linewidth = .5)
        
        ### Get mean neural push for each task ###
        segment_mn = np.mean(push[ixi, :], axis=0)
        
        ### Plot this mean neural push as big black arrow ###
        ax_all[0].quiver(mag_mns[m]*np.cos(ang_mns[a]), mag_mns[m]*np.sin(ang_mns[a]), segment_mn[0], segment_mn[1], 
                  width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')
        
        ### ADJUSTMENT TO LEVEL 2 -- Now get the NEXT action commands ###
        ixip1 = ixi + 1; 
        
        ### If exceeds length after adding "1"
        if len(ixi) > 0:
            if ixip1[-1] > (len(push) -1):
                ixip1 = ixip1[:-1]
       
            ### Remove the commands that are equal to the minimum bin_cnt (=1) 
            ### because they spilled over from the last trial ###
            kp0 = np.nonzero(bin_num[ixip1] > np.min(bin_num))[0]
            
            ### Keep these guys ###
            index_og = ixip1[kp0]
            
            ### Arrows dict 
            ### We need to create this because we're not sure how big to maek the colormap when plotting; 
            arrows_dict = []        
            pred_arrows_dict = []

            ### Used to make the colorbar; 
            index_dict = []

            ### Used to make dA, dR colorbar; 
            dA_dict = []; 
            dR_dict = []; 
            
            ### Iterate through the tasks ###
            for mp in range(4):
                for ap in range(8):

                    ### LEVEL 2 -- OF THE NEXT ACTION COMMANDS, which match the action #####
                    index = np.nonzero(np.logical_and(command_bins[index_og, 0] == mp, command_bins[index_og, 1] == ap))[0]

                    if len(index) >= min_obs2:
                        print('Followup, m %d, a %d' %(mp, ap))
                        
                        ### Great plot this;
                        mn_next_action = np.mean(push[index_og[index], :], axis=0)
                        xi = mag_mns[mp]*np.cos(ang_mns[ap])
                        yi = mag_mns[mp]*np.sin(ang_mns[ap])
                        vx = mn_next_action[0];
                        vy = mn_next_action[1]; 

                        arrows_dict.append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(vx), copy.deepcopy(vy), len(index)])
                        index_dict.append(len(index))
                        
                        #### Predicted Action ####
                        if pred_push is None:
                            partsss
                        else:
                            pred_mn_next_action = np.mean(pred_push[index_og[index], :], axis=0)
                            p_vx = pred_mn_next_action[0]
                            p_vy = pred_mn_next_action[1]
                            
                            ### Get an estimate of the diff ang / diff mag from the reference; 
                            dA, dR = get_diffs(pred_mn_next_action, segment_mn)
                            
                            ### Add so we can do colorbar later; 
                            dA_dict.append(dA)
                            dR_dict.append(dR)

                            pred_arrows_dict.append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(p_vx), copy.deepcopy(p_vy), mp, ap, dA, dR])
            
            #import pdb; pdb.set_trace()
            ### Now figure out the color lists for CO / OBS separately; 
            if len(index_dict) > 0:
                num_corr = [0, 0] ## Total correct / total assessed; 

                #### Colors for distribution counts; ####
                mx_ = np.max(np.hstack((index_dict)))
                mn_ = np.min(np.hstack((index_dict)))
                cols = np.linspace(0., 1., mx_ - mn_ + 1)
                colors = [cm.viridis(x) for x in cols]
                print('All mx = %d, mn = %d' %(mx_, mn_))
            
                ### Get colors for dA / dR; 
                mxa_ = np.max(np.hstack((dA_dict)))
                mna_ = np.min(np.hstack((dA_dict)))

                mxr_ = np.max(np.hstack((dR_dict)))
                mnr_ = np.min(np.hstack((dR_dict)))
                
                npts = 5; 

                ### Color dict ###
                cols = np.linspace(0., 1., npts)
                sub_colors = [cm.viridis(x) for x in cols]
                
                ### These are the center colors for each of the segments; 
                ang_dict = np.linspace(mna_, mxa_, npts)
                mag_dict = np.linspace(mnr_, mxr_, npts)

                for arrow in arrows_dict:
                    
                    ### Parse the parts 
                    xi, yi, vx, vy, N = arrow; 
                    
                    ### Plot it;
                    ax_all[0].quiver(xi, yi, vx, vy,
                                      width = arrow_scale, color = colors[N - mn_], 
                                        angles='xy', scale=1, scale_units='xy')
                
                for pred_arrow in pred_arrows_dict:
                    
                    ### Parse the parts
                    pxi, pyi, pvx, pvy, mi, ai, da, dr = pred_arrow
                    
                    ### Plot the arrows in "outline / dashd red" if they are correct
                    ### Plot them in "filled in alpha red" if they are wrong; 
                    corr_dir = assess_corr_dir([m, a], [mi, ai], segment_mn, np.array([pvx, pvy]))

                    if corr_dir is None: 
                        ax_all[0].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='gray', facecolor='none', 
                                          linewidth = 1.)

                    elif corr_dir == True: 
                        ### Plot it; 
                        num_corr[0] += 1
                        num_corr[1] += 1
                        
                        ax_all[0].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='r', facecolor='none', 
                                          linewidth = 1.)


                    elif corr_dir == False: 
                        num_corr[1] += 1
                        ### Plot it; 
                        ax_all[0].quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          facecolor='r', alpha = 0.4, linewidth=0.)

                    ######## Add info on the new plots ########
                    if add_ang_mag_subplots:

                        ### Get the axes 
                        ang_ax = ax_all[1]; 
                        mag_ax = ax_all[2]; 

                        ### Plot a scatter? 
                        x_pts, y_pts = get_pts_in_sector(mi, ai, mag_bound)

                        ### Which color ix is da and dr; 
                        ang_col_ix = np.argmin(np.abs(ang_dict - da))
                        mag_col_ix = np.argmin(np.abs(mag_dict - dr))

                        ax_all[1].plot(x_pts, y_pts, '.', color = sub_colors[ang_col_ix])
                        ax_all[2].plot(x_pts, y_pts, '.', color = sub_colors[mag_col_ix])

                for ax in ax_all:
                    ax.set_xlim([-lims, lims])
                    ax.set_ylim([-lims, lims])
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                    if num_corr[1] == 0:
                        pass
                    else:
                        pc = int(100*(float(num_corr[0])/float(num_corr[1])))
                        ax.set_title('Mag %d, Ang %d, Perc Corr: %d' %( m, a, pc))

                ##################################################################
                ########### Set colorbar for the main color plot #################
                ##################################################################
                cmap = mpl.cm.viridis
                
                if mn_ == mx_:
                    norm = mpl.colors.Normalize(vmin=mn_, vmax=mx_+.1)
                else:
                    norm = mpl.colors.Normalize(vmin=mn_, vmax=mx_)
    
                if add_ang_mag_subplots:
                    cax0 = fig.add_axes([0.15, 0.1, 0.2, 0.05])
                else:
                    cax0 = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    
                cb1 = mpl.colorbar.ColorbarBase(cax0, cmap=cmap,
                                    norm=norm, orientation='horizontal',
                                    boundaries = np.arange(mn_-0.5, mx_+1.5, 1.))
                tmp2 = np.mean([mn_, mx_+1])
                cb1.set_ticks([mn_, tmp2, mx_+1]) 
                cb1.set_ticklabels([mn_, tmp2, mx_+1])
                cb1.set_label('Counts')

                ##################################################################
                ########### Add colorbar for the main color plot #################
                ##################################################################
                if add_ang_mag_subplots:

                    cax1 = fig.add_axes([0.4, 0.1, 0.2, 0.05])
                    cax2 = fig.add_axes([0.7, 0.1, 0.2, 0.05])
                
                    norm = mpl.colors.Normalize(vmin=mna_/np.pi*180, vmax=mxa_/np.pi*180)
                    cb2 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap,
                                        norm=norm, orientation='horizontal')
                    cb2.set_label('dAngle')
                    angs = np.array([mna_, np.mean([mna_, mxa_]), mxa_])/np.pi*180
                    angs[0] += 2
                    angs[-1] -= 2
                    cb2.set_ticks(angs.astype(int))
                    cb2.set_ticklabels(angs.astype(int))


                    norm = mpl.colors.Normalize(vmin=mnr_, vmax=mxr_)
                    cb3 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap,
                                        norm=norm, orientation='horizontal')
                    cb3.set_label('dMag')
                    rads = np.array([mnr_, np.mean([mnr_, mxr_]), mxr_])
                    tmp_dr = 1/20.*(mxr_ - mnr_)
                    rads[0] += tmp_dr 
                    rads[-1] -= tmp_dr
                    rads = np.round(100.*rads).astype(int) 
                    rads = rads/100.
                    cb3.set_ticks(rads)
                    cb3.set_ticklabels(rads)

                ### Save this ? 
                if save:
                    fig.savefig(save_dir+'/'+prefix+'_'+title_string+'.png')

def plot_real_vs_pred_dAngle_dMag(command_bins=None, push=None, bin_num=None, mag_bound=None, trg=None, 
    tsk=None, pred_push=None, min_obs = 15, save=False, save_dir = None, prefix=None, shuff_data = False, **kwargs):
    '''
    This is a method to plot real change in angle/mag from current bin to next bin for cases where 
    there are at least 15 observations of each
    '''

    #f, ax = plt.subplots(ncols = 2, figsize = (8, 4))
    
    reg_dict = dict(dA = [], pred_dA = defaultdict(list), dM = [], pred_dM = defaultdict(list))

    #### Percent Correct ####
    perc_corr = dict(true = defaultdict(list), cw = defaultdict(list), ccw = defaultdict(list))
    perc_corr['ang_pairwise'] = defaultdict(list)
    perc_corr['mag'] = defaultdict(list)

    if shuff_data:
        nShuff = pred_push.shape[2]
    else:
        nShuff = 1;
        pred_push = pred_push[:, :, np.newaxis]

    for m in range(4):
        for a in range(8):
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]

            if len(ix) >= min_obs:
                ixp1 = ix + 1; 
                ixp1 = ixp1[ixp1 < len(command_bins)]
                ixp1 = ixp1[bin_num[ixp1] > np.min(bin_num)]

                mag_dict = dict(id=[])

                for mp in range(4):
                    for ap in range(8):
                        ix2 = np.nonzero(np.logical_and(command_bins[ixp1, 0] == mp, command_bins[ixp1, 1] == ap))[0]
                        ix_tp1 = ixp1[ix2]

                        if len(ix_tp1) >= min_obs:

                            ### Then compute the mean vectors; 
                            segment_mn = np.mean(push[ix, :], axis=0)
                            next_segment_mn = np.mean(push[ix_tp1, :], axis=0)
                            
                            #if shuff_data:
                            pred_next_segment_mn = np.mean(pred_push[ix_tp1, :, :], axis=0)
                            assert(pred_next_segment_mn.shape[1] == nShuff)
                            mag_dict[mp, ap] = np.linalg.norm(pred_next_segment_mn, axis=0)
                            
                            mag_dict[mp, ap, 'full'] = pred_next_segment_mn 
                            mag_dict['id'].append([mp, ap])

                            ### Get angle / magnitude diffs; 
                            ### Angle is unsigned; 
                            dA, dM = get_diffs(next_segment_mn, segment_mn)
                            reg_dict['dM'].append(dM)

                            ### Sign the angle; 
                            ### As convention, we'll call CW (-) and CCW (+)
                            cp = np.sign(np.cross(segment_mn, next_segment_mn))
                            dA = cp*dA
                            reg_dict['dA'].append(dA/np.pi*180)

                            ### Now get the predicted value; 
                            ###### Assess the first shuffle 
                            for ns in range(nShuff):                                    
                                pred_dAi, pred_dMi = get_diffs(pred_next_segment_mn[:, ns], segment_mn)
                                pred_cp = np.sign(np.cross(segment_mn, pred_next_segment_mn[:, ns]))
                                reg_dict['pred_dA'][ns].append(pred_cp*np.abs(pred_dAi)/np.pi*180)
                                reg_dict['pred_dM'][ns].append(pred_dMi)
                            
                            #ax[0].plot(dA/np.pi*180, pred_dA/np.pi*180, 'k.')
                            #ax[1].plot(dM, pred_dM, 'k.')
                            ######### Save these #######

                            ### Figure out if we should assess precent correct of magnitude 
                            for ns in range(nShuff):
                                corr_dir = assess_corr_dir([m, a], [mp, ap], segment_mn, pred_next_segment_mn[:, ns])

                                if corr_dir is None:
                                    pass
                                elif corr_dir == True: 
                                    perc_corr['true'][ns].append(1.)
                                elif corr_dir == False:
                                    perc_corr['true'][ns].append(0.)

                                ### Now want to see chance level perc correct if next action was always CW or always CCW
                                # if corr_dir is not None:
                                #     ### Is this division CW or CCW: 
                                #     ## cp is the sign, CW / CCW; 
                                #     if cp == -1: 
                                #         perc_corr['cw'][0] += 1
                                #     elif cp == 1:
                                #         perc_corr['ccw'][0] += 1

                                #     perc_corr['cw'][1] += 1
                                #     perc_corr['ccw'][1] += 1 
                    
                ### Ok now done with all these 
                if len(mag_dict['id']) > 1:
                    tmp_ids = np.vstack((mag_dict['id'])) 

                    ##### For each angle see if magnitudes are properly aligned 
                    for tmp_a in range(8):
                        ix_a = np.nonzero(tmp_ids[:, 1] == tmp_a)[0]

                        ### If more than 1 of this angle; 
                        if len(ix_a) > 1:
                            tmp_n = len(ix_a)

                            ### Comparisons; 
                            for i_t0 in range(tmp_n):
                                tmp_mg0 = tmp_ids[ix_a[i_t0], 0]

                                ### This is norm x nShuff 
                                tmp_norm0 = mag_dict[tmp_mg0, tmp_a]

                                for i_t1 in range(i_t0+1, tmp_n):
                                    tmp_mg1 = tmp_ids[ix_a[i_t1], 0]
                                    assert(tmp_mg1 != tmp_mg0)

                                    #### This is norm x nShuff; 
                                    tmp_norm1 = mag_dict[tmp_mg1, tmp_a]

                                    #### For each shuffle do this comparison 
                                    for ns in range(nShuff):
                                        dtmp = float(np.sign(tmp_mg1 - tmp_mg0))
                                        dnorm = float(np.sign(tmp_norm1[ns] - tmp_norm0[ns]))
                                        
                                        if dtmp == dnorm:
                                            perc_corr['mag'][ns].append(1.)
                                        else:
                                            perc_corr['mag'][ns].append(0.)

                    #### Structure within the angles themselves ####
                    for tmp_m in range(4):

                        #### For each magnitude see if angles are properly aligned #####
                        ix_m = np.nonzero(tmp_ids[:, 0] == tmp_m)[0]
                        
                        if len(ix_m) > 1:
                            tmp_n = len(ix_m)

                            for i_t0 in range(tmp_n):
                                tmp_ag0 = tmp_ids[ix_m[i_t0], 1]
                                tmp_pred0 = mag_dict[tmp_m, tmp_ag0, 'full']

                                for i_t1 in range(i_t0+1, tmp_n):
                                    tmp_ag1 = tmp_ids[ix_m[i_t1], 1]
                                    tmp_pred1 = mag_dict[tmp_m, tmp_ag1, 'full']

                                    ### Make sure not the same angle ### 
                                    assert(tmp_ag1 != tmp_ag0)

                                    for ns in range(nShuff):
                                        ##### Assess if starting from i_t0 and transitioning to i_t1, if angles move in the right direciton; 
                                        # assess_corr_dir(disc_dir, disc_pred_dir, mn_dir, mn_pred_dir)
                                        corr_dir = assess_corr_dir([tmp_m, tmp_ag0], [tmp_m, tmp_ag1], tmp_pred0[:, ns], tmp_pred1[:, ns])
                                        assert(corr_dir is not None)
                                        if corr_dir == True: 
                                            perc_corr['ang_pairwise'][ns].append(1.)
                                        else:
                                            perc_corr['ang_pairwise'][ns].append(0.)

    
    #ax[0].plot([-150, 150], [-150, 150], 'k-')
    #ax[1].plot([-3, 1], [-3, 1], 'k-')
    
    R2_dA = []
    for ns in range(nShuff):
        slp,intc,rv,pv,stedrr = scipy.stats.linregress(np.hstack((reg_dict['dA'])), np.hstack((reg_dict['pred_dA'][ns])))
        R2_dA.append(util_fcns.get_R2(np.hstack((reg_dict['dA'])), np.hstack((reg_dict['pred_dA'][ns]))))

    #ax[0].set_title('Slp: %.2f, Rval: %.2f, R2-err: %.2f' %(slp, rv, r2))
    #xtmp = np.linspace(-150, 150, 10)
    #ytmp = slp*xtmp + intc
    # ax[0].plot(xtmp, ytmp, 'r--')
    # ax[0].set_xlabel('True Angle Change (deg)')
    # ax[0].set_ylabel('Pred Angle Change (deg)')
    #ang_stats = [slp, rv, r2]

    R2_dM = []
    for ns in range(nShuff):
        slp,intc,rv,pv,stedrr = scipy.stats.linregress(np.hstack((reg_dict['dM'])), np.hstack((reg_dict['pred_dM'][ns])))
        R2_dM.append(util_fcns.get_R2(np.hstack((reg_dict['dM'])), np.hstack((reg_dict['pred_dM'][ns]))))

    #ax[1].set_title('Slp: %.2f, Rval: %.2f, R2-err: %.2f' %(slp, rv, r2))
    #xtmp = np.linspace(-3, 1, 10)
    #ytmp = slp*xtmp + intc
    # ax[1].plot(xtmp, ytmp, 'r--')
    # ax[1].set_xlabel('True Mag Change')
    # ax[1].set_ylabel('Pred Mag Change')
    #f.tight_layout()
    #mag_stats = [slp, rv, r2]
 
    # if save:
    #     f.savefig(save_dir + prefix+'.png')
    return R2_dA, R2_dM, perc_corr

def plot_dAngle_dMag_xcond(command_bins=None, push=None, bin_num=None, mag_bound=None, trg=None, 
    tsk=None, pred_push=None, min_obs = 15, save=False, save_dir = None, prefix = None, **kwargs):
    '''
    method to take command bins for a specific task/target where there are more than min_obs
    then take the next action and predicted next action
    '''

    f, ax = plt.subplots(ncols = 2, figsize = (8, 4))
    reg_dict = dict(d_dA = [], pred_d_dA = [], d_dM = [], pred_d_dM = [])

    for tg0 in range(10):
        ix_tg0 = np.nonzero(np.logical_and(tsk == 0, trg == tg0))[0]

        for tg1 in range(10):
            ix_tg1 = np.nonzero(np.logical_and(tsk == 1, trg == tg1))[0]

            for m in range(4):
                for a in range(8):

                    ### Make sure there are enough of this bin in BOTH conditions; 
                    com_ix0 = np.nonzero(np.logical_and(command_bins[ix_tg0, 0] == m, command_bins[ix_tg0, 1] == a))[0]
                    com_ix1 = np.nonzero(np.logical_and(command_bins[ix_tg1, 0] == m, command_bins[ix_tg1, 1] == a))[0]

                    ### Great! Plot this comparison ###
                    com_ix0_tp1 = com_ix0 + 1
                    com_ix1_tp1 = com_ix1 + 1

                    ### Remove the long ones
                    com_ix0_tp1 = com_ix0_tp1[com_ix0_tp1 < len(command_bins)]
                    com_ix1_tp1 = com_ix1_tp1[com_ix1_tp1 < len(command_bins)]

                    ### Remove the ones that cross over trials 
                    com_ix0_tp1 = com_ix0_tp1[bin_num[com_ix0_tp1] > np.min(bin_num)]
                    com_ix1_tp1 = com_ix1_tp1[bin_num[com_ix1_tp1] > np.min(bin_num)]

                    ### Re-do so now the same length
                    com_ix0 = com_ix0_tp1 - 1; 
                    com_ix1 = com_ix1_tp1 - 1; 

                    if np.logical_and(len(com_ix0_tp1) >= min_obs, len(com_ix1_tp1) >= min_obs):

                        ### Average segment ####
                        mn_seg0 = np.mean(push[com_ix0, :], axis=0)
                        mn_seg1 = np.mean(push[com_ix1, :], axis=0)

                        ### Average next segment ####
                        mn_seg0_tp1 = np.mean(push[com_ix0_tp1, :], axis=0)
                        mn_seg1_tp1 = np.mean(push[com_ix1_tp1, :], axis=0)

                        ### Average predicted next segment; 
                        pred_mn_seg0_tp1 = np.mean(pred_push[com_ix0_tp1, :], axis=0)
                        pred_mn_seg1_tp1 = np.mean(pred_push[com_ix1_tp1, :], axis=0)

                        ### Angle differences ###
                        ### What are the across condition difference ###
                        ### Diffs of Diffs ### 
                        dA_0, dM_0 = get_diffs(mn_seg0_tp1, mn_seg0)
                        dA_1, dM_1 = get_diffs(mn_seg1_tp1, mn_seg1)
                        
                        pdA_0, pdM_0 = get_diffs(pred_mn_seg0_tp1, mn_seg0)
                        pdA_1, pdM_1 = get_diffs(pred_mn_seg1_tp1, mn_seg1)

                        ### Get the CW / CCW part; 
                        dA_0 = np.sign(np.cross(mn_seg0, mn_seg0_tp1))*dA_0
                        dA_1 = np.sign(np.cross(mn_seg1, mn_seg1_tp1))*dA_1

                        pdA_0 = np.sign(np.cross(mn_seg0, pred_mn_seg0_tp1))*pdA_0
                        pdA_1 = np.sign(np.cross(mn_seg1, pred_mn_seg1_tp1))*pdA_1
                        
                        ### Now get the differences in angle; 
                        d_dA = dA_0 - dA_1; 
                        d_pdA = pdA_0 - pdA_1; 

                        ### Now do the same thing with magnitudes;
                        d_dM = dM_0 - dM_1; 
                        d_pdM = pdM_0 - pdM_1; 

                        if tg1 == tg0 + 1:
                            col = 'r'
                        else:   
                            col = 'k'

                        ax[0].plot(r2a(d_dA), r2a(d_pdA), '.', color=col)
                        ax[1].plot(d_dM, d_pdM, '.', color=col)

                        reg_dict['d_dA'].append(r2a(d_dA))
                        reg_dict['pred_d_dA'].append(r2a(d_pdA))

                        reg_dict['d_dM'].append(d_dM)
                        reg_dict['pred_d_dM'].append(d_pdM)

    ax[0].set_xlabel('Ang Diffs in (True Next - Current)\n Action Across Cond (ang)')
    ax[0].set_ylabel('Ang Diffs in (Pred Next - Current)\n Action Across Cond (ang)')
    ax[1].set_xlabel('Mag Diffs in (True Next - Current)\n Action Across Cond')
    ax[1].set_ylabel('Mag Diffs in (Pred Next - Current)\n Action Across Cond')

    ax[0].plot([-150, 150], [-150, 150], 'k-')
    ax[1].plot([-3, 1], [-3, 1], 'k-')

    reg_dict = util_fcns.hstack_keys(reg_dict)

    ### Plot predictions #### 
    slp,intc,rv,pv,stedrr = scipy.stats.linregress(reg_dict['d_dA'], reg_dict['pred_d_dA'])
    r2 = util_fcns.get_R2(reg_dict['d_dA'], reg_dict['pred_d_dA'])
    ax[0].set_title('Slp: %.2f, Rval: %.2f, R2 err: %.2f' %(slp, rv, r2), color = 'b')
    xtmp = np.linspace(-150, 150, 10)
    ytmp = slp*xtmp + intc
    ax[0].plot(xtmp, ytmp, 'b--')
    ang_stats = [slp, rv, r2] 

    slp,intc,rv,pv,stedrr = scipy.stats.linregress(reg_dict['d_dM'], reg_dict['pred_d_dM'])
    r2 = util_fcns.get_R2(reg_dict['d_dM'], reg_dict['pred_d_dM'])
    ax[1].set_title('Slp: %.2f, Rval: %.2f, R2-err: %.2f' %(slp, rv, r2), color = 'b')
    xtmp = np.linspace(-3, 1, 10)
    ytmp = slp*xtmp + intc
    ax[1].plot(xtmp, ytmp, 'b--')
    f.tight_layout()
    mag_stats = [slp, rv, r2]
    if save:
        f.savefig(save_dir + prefix+'.png')

    return ang_stats, mag_stats

def r2a(rad):
    return rad/np.pi*180.

############ Preproc #######
def preproc(animal, model_set_number, dyn_model, day, model_type = 2, minobs = 10, minobs2 = 3,
    only_potent = False, include_shuffle = False, dat = None):
    ''' 
    method for preprocessing/extracting data from tuning model files before doing all the scatter plots 
    model_set_number and dyn_model dictate which model is used; 
    model_type: general (2) or condition specific ('cond') are options; 
    minobs/minobs2 are used to set prefix; 
    only_potent is used to indicate that only potent neural data should be used to make predictions 
        at the next time step; this involves using the same model, but preprocessing the data used to predict 
        in order to keep only the potent parts; 
    include_shuffle: adds shuffled to the data set processed; 
        -- edit on 7/2/20 --> changed shuffle to load data from the 100 shuffles 
    '''

    if animal == 'grom':
        _, K = util_fcns.get_grom_decoder(day)
    else:
        K = util_fcns.get_jeev_decoder(day)

    ### Open the model type ####
    if model_type in [0, 1, 2]:
        ### Task spec vs. general
        if dat is None:     
            dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(model_set_number), 'rb'))
            
        # if include_shuffle:
        #     dat_shuff = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_within_bin_shuff0.pkl' %(model_set_number), 'rb'))
            
    elif model_type == -1: 
        ### Full model 
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d.pkl' %(model_set_number), 'rb'))

    elif model_type == 'cond':
        ### Condition specific models; 
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_cond_spec.pkl' %(model_set_number), 'rb'))
        ndays = analysis_config.data_params[animal+'_ndays']
        dat_reconst = generate_models_utils.reconst_spks_from_cond_spec_model(dat, dyn_model, ndays)

    if dyn_model == 'hist_1pos_0psh_0spksm_1_spksp_0':
        prefix = 'day%d_minobs_%d_%d_modelset%d'%(day, minobs, minobs2, model_set_number)
        
    elif dyn_model == 'hist_1pos_0psh_0spksm_1_spksp_1':
        prefix = 'pred_w_hist_and_fut_day%d_minobs_%d_%d_modelset%d'%(day, minobs, minobs2, model_set_number)
    
    else:
        raise Excpetion('no other models yet')

    if model_type == 2:
        prefix = prefix + '_gen'
    elif model_type == 'cond':
        prefix = prefix + '_cond'
    else:
        raise Exception('no other model types yet')

    if only_potent:
        prefix = prefix + '_pred_w_pot'

    ####### Get out only potent #######
    if model_type in [0, 1, 2]:
        if only_potent:
            raise Exception('deprecated')
            pred_spks = dat[day, dyn_model, 'pot'][:, :, model_type]
            # if include_shuffle:
            #     pred_spks_shuff = dat_shuff[day, dyn_model, 'pot'][:, :, model_type]
        else:
            pred_spks = dat[day, dyn_model][:, :, model_type]
            if include_shuffle:
                pred_spks_shuff = plot_generated_models.get_shuffled_data(animal, day, dyn_model) #dat_shuff[day, dyn_model][:, :, model_type]
                nShuff = pred_spks_shuff.shape[2]

    elif model_type == 'cond':
        if only_potent:
            pred_spks = dat_reconst[day, dyn_model, 'pot']
        else:
            pred_spks = dat_reconst[day, dyn_model]

        if include_shuffle:
            raise Exception('not yet implemented')

    #### True spks #####
    spks = dat[day, 'spks']

    ### Rest of stuff ####
    if animal == 'grom':
        pred_push = np.dot(K[[3, 5], :], pred_spks.T).T
        if include_shuffle:
            pred_push_shuff = []
            for ns in range(nShuff):
                pred_push_shuff.append(np.dot(K[[3, 5], :], pred_spks_shuff[:, :, ns].T).T)
            pred_push_shuff = np.dstack((pred_push_shuff))
            assert(pred_push_shuff.shape[2] == nShuff)

    elif animal == 'jeev':
        pred_push = np.dot(K, pred_spks.T).T
        if include_shuffle:
            pred_push_shuff = []
            for ns in range(nShuff):
                pred_push_shuff.append(np.dot(K, pred_spks_shuff[:, :, ns].T).T)
            pred_push_shuff = np.dstack((pred_push_shuff))
            assert(pred_push_shuff.shape[2] == nShuff)

    bin_num = dat[day, 'bin_num']
    tsk = dat[day, 'task']
    pos = dat[day, 'pos']
    vel = dat[day, 'vel']
    trg = dat[day, 'trg']
    push = dat[day, 'np']

    assert(pred_push.shape == push.shape)
    assert(len(tsk) == len(trg) == len(pos) == len(vel) == len(push) == len(bin_num))
    
    if include_shuffle:
        assert(spks.shape[0] == pred_spks_shuff.shape[0] == pred_spks.shape[0])
        assert(spks.shape[1] == pred_spks_shuff.shape[1] == pred_spks.shape[1])

    ### Print the overall R2 of the push; 
    R2 = util_fcns.get_R2(push, pred_push)
    print('R2 of action %.2f' %(R2))

    R2 = util_fcns.get_R2(spks, pred_spks)
    print('R2 of neural %.2f' %(R2))
    
    ### Segment up the pushes into discrete bins ###
    mag_boundaries = pickle.load(open(analysis_config.config['grom_pref'] + 'radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    command_bins = util_fcns.commands2bins([push], mag_boundaries, animal, day, vel_ix = [0,1], ndiv=8)[0]
    
    data = dict(spks = spks, pred_spks = pred_spks, pred_push = pred_push, bin_num = bin_num, tsk = tsk, pos = pos, vel = vel, trg = trg, push = push,
        mag_bound = mag_boundaries[animal, day], command_bins = command_bins, prefix = prefix)
    if include_shuffle:
        data['pred_spks_shuff'] = pred_spks_shuff
        data['pred_push_shuff'] = pred_push_shuff

    return data 

############ UTILS ###########

def assess_corr_dir(disc_dir, disc_pred_dir, mn_dir, mn_pred_dir):
    m, a = disc_dir
    mi, ai = disc_pred_dir

    if a ==  ai:
        return None
    else:

        ang_mns = np.linspace(0, 2*np.pi, 9)

        ### Center angle ####
        disc_ang = np.array([np.cos(ang_mns[a]), np.sin(ang_mns[a])])

        ### Predicted angle ###
        disc_ang_pred = np.array([np.cos(ang_mns[ai]), np.sin(ang_mns[ai])])

        ## Ok now do the cross product to see if CW (-1) or CCW (+1)
        true_cp = np.sign(np.cross(disc_ang, disc_ang_pred))

        ### Now do the predicted 
        act_cp = np.sign(np.cross(mn_dir, mn_pred_dir))

        return true_cp == act_cp

def get_pts_in_sector(m, a, mag_bound, npts=100):

    ### Add the start / end ###
    tmp = np.hstack(([0], mag_bound, [mag_bound[2] + 2]))

    ### Get the 2 radii
    r1 = tmp[m]
    r2 = tmp[m+1]
    dr = r2 - r1; 

    ### Angles 
    tmp2 = np.linspace(-np.pi/8., (2*np.pi) - np.pi/8., 9)
    a1 = tmp2[a]
    a2 = tmp2[a+1]
    da = a2 - a1; 

    ### Randomly sample b/w 
    rad_pts = np.random.rand(npts, )*dr + r1; 
    ang_pts = np.random.rand(npts, )*da + a1; 

    x1 = rad_pts*np.cos(ang_pts)
    y1 = rad_pts*np.sin(ang_pts)

    return x1, y1

def get_diffs(pred_mn_next_action, segment_mn):

    nm = np.linalg.norm(segment_mn)
    nm2 = np.linalg.norm(pred_mn_next_action)

    dM = nm2 - nm # Positiv if nm2 > nm, neg if nm2 < nm; 
    dA = np.arccos(np.dot(pred_mn_next_action / nm2, segment_mn / nm))

    return dA, dM

def _test_get_diffs():
    for i in range(1000):
        a1 = np.random.rand()*2*np.pi
        a2 = np.random.rand()*2*np.pi
        dAr = np.abs(a2-a1)
        if dAr > np.pi: 
            dAr = 2*np.pi - dAr

        r1 = np.random.rand()*5
        r2 = np.random.rand()*5
        dRr = r1 - r2

        v1 = r1*np.array([np.cos(a1), np.sin(a1)])
        v2 = r2*np.array([np.cos(a2), np.sin(a2)])

        dA, dM = get_diffs(v1, v2)
        assert(np.allclose(dA, dAr))
        assert(np.allclose(dM, dRr))
    
    
    