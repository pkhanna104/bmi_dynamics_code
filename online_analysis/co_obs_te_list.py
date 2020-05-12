import numpy as np
import matplotlib.pyplot as plt
import tables
import pickle
import scipy.stats
import math
import glob
import os, basic_hdf_analysis
import pandas

import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5, style='white')


#########################################
#### Start 3/2 #############
#########################################

## TE NUM, ## SIGNAL_TYPE (TASK_SIGNAL), ## DECODER TRAINING TASK ## DATE
co_all = [[4377, 4391, 4395, 4411, 4499, 4510, 4523, 4525, 4536, 4553, 4558, 4585, 4586, 4587, 4600, 4601, 4613, 4619], 
          ['co_all', 'co_all','co_all', 'co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all','co_all', 'co_all', 'co_all'], 
          ['co', 'co', 'co','co','co','obs', 'obs','obs','obs','obs','obs','co','co','co','co','co','co','co', 'co'], 
          ['3-2', '3-4', '3-4', '3-7', '3-15', '3-16','3-17','3-17', '3-18','3-19', '3-19','3-21','3-21','3-21','3-22', '3-22', '3-23', '3-24']]

co_co_shar = [[4512, 4604, 4616, 4627], 
              ['co_w_co_mn_sc_shar', 'co_w_co_sc_shar','co_w_co_mn_sc_shar','co_w_co_mn_sc_shar'], 
              ['obs', 'co','co', 'co'], 
              ['3-16', '3-22', '3-23', '3-24']]

co_obs_shar = [[4503, 4511, 4513,4524, 4554], 
                ['co_w_obs_mn_sc_shar','co_w_obs_mn_sc_shar', 'co_w_obs_sc_shar','co_w_obs_mn_sc_shar', 'co_w_obs_mn_sc_shar'], 
                ['obs','obs', 'obs','obs','obs'], 
                ['3-15', '3-16','3-16','3-17', '3-19']]

obs_all = [[4378, 4392, 4394, 4412, 4415, 4497, 4498, 4504, 4509, 4514, 4520, 4522, 4526, 4532, 4533, 4549, 4560], 
            ['obs_all', 'obs_all', 'obs_all','obs_all', 'obs_all','obs_all','obs_all','obs_all','obs_all','obs_all','obs_all','obs_all','obs_all','obs_all','obs_all', 'obs_all','obs_all'], 
            ['co', 'co', 'co', 'co', 'co', 'obs', 'obs', 'obs','obs','obs','obs','obs','obs','obs','obs','obs','co'], 
            ['3-2', '3-4','3-4', '3-7','3-7', '3-15', '3-15', '3-15','3-16','3-16', '3-17', '3-17','3-17','3-18','3-18','3-19','3-19']]


obs_co_shar = [[4396, 4559, 4384], 
               ['obs_w_co_mn_sc_shar','obs_w_co_mn_sc_shar', 'obs_w_co_mn_sc_shar'], 
               ['co','co', 'co'], 
               ['3-4','3-19', '3-2']]

obs_obs_shar = [[4505, 4515, 4521, 4552],
                ['obs_w_obs_mn_sc_shar','obs_w_obs_mn_sc_shar','obs_w_obs_mn_sc_shar','obs_w_obs_mn_sc_shar'],
                ['obs','obs','obs','obs'],
                ['3-15','3-16','3-17','3-19']
                ]

poor_data = [4515, 4384, 4505]
data_dict = dict( (name,eval(name)) for name in ['co_all', 'co_co_shar', 'co_obs_shar', 'obs_all', 'obs_co_shar', 'obs_obs_shar', 'poor_data'])

task_name = dict(co_all='bmi_resetting', co_co_shar='fa_bmi', co_obs_shar='fa_bmi',obs_all='bmi_resetting_w_obstacles', 
    obs_co_shar='fa_bmi_w_obs',obs_obs_shar='fa_bmi_w_obs')
input_names =['main_sc_shared', 'shared_scaled']
trained_from_names = dict(co_co_shar='bmi_resetting', co_obs_shar='bmi_resetting_w_obstacles', obs_obs_shar='bmi_resetting_w_obstacles', obs_co_shar='bmi_resetting')

def check_data(data_dict, task_name, input_names, trained_from_names):
    ##############################
    #### Check Data #############
    ##############################
    for i_d, dat in enumerate(data_dict.keys()):
      if dat is not 'poor_data':
          for i_t, te in enumerate(data_dict[dat][0]):
              ten = dbfn.TaskEntry(te)
                
              # Correct task:
              assert ten.task.name == task_name[dat]

              # Correct inputs:
              if 'shar' in dat:
                  assert ten.hdf.root.task.attrs.input_type in input_names

                  #FA trained from correct task: 
                  try:
                      te_train = dbfn.TaskEntry(ten.hdf.root.fa_params.training_task_entry[0])
                      assert te_train.task.name == trained_from_names[dat]
                  except:
                      print te, dat, 'no fa params'


########################################
### Make a HDF Performance tables ######
########################################
def make_input_type(data_dict=data_dict):
    input_type = {}
    for i, k in enumerate(data_dict.keys()):
        if k is not 'poor_data':
            x = data_dict[k]
            for it, te in enumerate(x[0]):
                if te not in data_dict['poor_data']:
                    input_type[te] = [x[1][it], k, x[2][it], x[3][it]]
                else:
                    print 'rejecting: ', te, ' due to inclusion in poor data list!'
    return input_type

def make_hdf_file(data_dict=data_dict, suffix=''):
    te_list = np.hstack((v[0] for i, (k, v) in enumerate(data_dict.items()) if k != 'poor_data'))
    new_hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2017_performance_w_shar_co_obs_on_tasks_co_obs_include_control_'+suffix+'.hdf')
    basic_hdf_analysis.process_targets(te_list, new_hdf_name)

def make_dataframe(input_type, total_or_share='share', norm='sub'):
    # t2t normalizing factor is 

    ###########################
    ### Make a Dataframe ######
    ###########################
    # try:
    #     hdf = tables.openFile('/Volumes/TimeMachineBackups/grom2016/grom2017_performance_w_shar_co_obs_on_tasks_co_obs_include_control_w_rpm_table.hdf')
    # except:
    #     hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2017_performance_w_shar_co_obs_on_tasks_co_obs_include_control_w_rpm_table.hdf')
    
    # Edit as of 2-3-19: fixed path error measurement so sum of absolute value of angular error, not sum of error
    #hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2017_performance_w_shar_co_obs_on_tasks_co_obs_include_control_fix_path_error.hdf')
    hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2017_performance_w_shar_co_obs_on_tasks_co_obs_include_control_add_eff.hdf')
    avg_t2t = {}
    avg_pe = {}
    avg_avg_pe = {}
    avg_pl = {}
    avg_sp = {}
    avg_d2t = {}

    ### Calculate the average metric for each input type (co_all or obs_all) and each target ###
    for j in range(len(hdf.root.trial_metrics)):
        i = hdf.root.trial_metrics[j]
        if i['task_entry'] in input_type.keys():
            if input_type[i['task_entry']][1] in ['co_all', 'obs_all']:
                typ = input_type[i['task_entry']][1]
                dat = input_type[i['task_entry']][3]
                try:
                    #avg_t2t[dat, i['target_index']].append(i['time2targ'])
                    avg_t2t[typ, dat, i['target_index']].append(i['time2targ'])
                    avg_pe[typ, dat, i['target_index']].append(i['path_error'])
                    avg_avg_pe[typ, dat, i['target_index']].append(i['avg_path_error'])
                    avg_pl[typ, dat, i['target_index']].append(i['path_length'])
                    avg_sp[typ, dat, i['target_index']].append(i['avg_speed'])
                    avg_d2t[typ, dat, i['target_index']].append(i['avg_dist2targ'])

                except:
                    #avg_t2t[dat, i['target_index']] = [i['time2targ']]
                    avg_t2t[typ, dat, i['target_index']] = [i['time2targ']]
                    avg_pe[typ, dat, i['target_index']] = [i['path_error']]
                    avg_pl[typ, dat, i['target_index']] = [i['path_length']]
                    avg_avg_pe[typ, dat, i['target_index']] = [i['avg_path_error']]
                    avg_sp[typ, dat, i['target_index']] = [i['avg_speed']]
                    avg_d2t[typ, dat, i['target_index']] = [i['avg_dist2targ']]
    avg_rpm = {}
    rpm = hdf.root.rpm[:]
    for j in range(len(rpm)):
        i = rpm[j]
        if i['task_entry'] in input_type.keys():
            if input_type[i['task_entry']][1] in ['co_all', 'obs_all']:
                typ = input_type[i['task_entry']][1]
                dat = input_type[i['task_entry']][3]
                try:
                    avg_rpm[typ, dat].append(i['trials_per_min_2min_chunks'])
                except:
                    avg_rpm[typ, dat] = [i['trials_per_min_2min_chunks']]


    ####################################
    ### Make Corrected t2t, pe, etc. ###
    ####################################

    df = {}
    for k in ['t2t', 't2t_corr', 'pe', 'pe_corr', 'pl', 'pl_corr', 'te', 'spd', 'cond_spec', 'cond_gen', 'date', 'dec', 'trl_num', 'ape',
        'ape_corr', 'spd_corr', 'avg_dt2', 'avg_dt2_corr']:
        df[k] = []

    df_rpm = {}
    for k in ['rpm', 'rpm_corr', 'te', 'cond_gen', 'dec', 'te_ord']:
        df_rpm[k] = []

    # df_met = {}
    # for k in ['perc_succ', 'cond_gen', 'date', 'dec']:
    #   df_met[k] = []

    tm = hdf.root.trial_metrics[:]
    for i, (t2t, pe, ape, pl, te, spd, rpm, ti, av_dt2) in enumerate(zip(tm['time2targ'], tm['path_error'], tm['avg_path_error'],
        tm['path_length'], tm['task_entry'], tm['avg_speed'], tm['rew_per_min'], tm['target_index'], tm['avg_dist2targ'])):

        if te in input_type.keys():

            date = input_type[te][3]
            cond_gen = input_type[te][1]

            if total_or_share == 'share':
                if cond_gen in ['co_all', 'obs_all']:
                    add = False
                else:
                    add = True
            elif total_or_share == 'total':
                if cond_gen in ['co_all', 'obs_all']:
                    add = True
                else:
                    add = False

            if add:
                # Which normalizer to use: 
                if cond_gen[:2] == 'co':
                    c = 'co_all'
                elif cond_gen[:3] == 'obs':
                    c = 'obs_all'

                av = np.mean(avg_t2t[c, date, ti])
                av_pe = np.mean(avg_pe[c, date, ti])
                av_av_pe = np.mean(avg_avg_pe[c, date, ti])
                av_pl = np.mean(avg_pl[c, date, ti])
                av_sp = np.mean(avg_sp[c, date, ti])
                av_dt2_mn = np.mean(avg_d2t[c, date, ti])
                
                if norm == 'sub':
                    t2t_corr = t2t - av
                    pe_corr = pe - av_pe
                    pl_corr = pl - av_pl
                    ape_corr = ape - av_av_pe
                    spd_corr = spd - av_sp
                    avg_dt2_corr = av_dt2 - av_dt2_mn

                elif norm == 'div':
                    t2t_corr = t2t / av; 
                    pe_corr = pe / av_pe
                    pl_corr = pl / av_pl
                    ape_corr = ape / av_av_pe
                    spd_corr = spd / av_sp
                    avg_dt2_corr = av_dt2 / av_dt2_mn

                df['t2t'].append(t2t)
                df['t2t_corr'].append(t2t_corr)
                df['pe'].append(pe)
                df['pe_corr'].append(pe_corr)
                df['ape'].append(ape)
                df['ape_corr'].append(ape_corr)
                df['pl'].append(pl)
                df['pl_corr'].append(pl_corr)
                df['avg_dt2'].append(av_dt2)
                df['avg_dt2_corr'].append(avg_dt2_corr)
                df['te'].append(te)
                df['spd'].append(spd)
                df['spd_corr'].append(spd_corr)
                df['cond_spec'].append(input_type[te][0])
                df['cond_gen'].append(input_type[te][1])
                df['dec'].append(input_type[te][2])
                df['date'].append(input_type[te][3])
                df['trl_num'].append(tm[i]['trial_number'])
    
    tm2 = hdf.root.rpm[:]
    tes_already = []
    te_cnt = 0

    for i, (te, rpm) in enumerate(zip(tm2['task_entry'], tm2['trials_per_min_2min_chunks'])):
        if te in tes_already:
            cnt += 1
        else:
            cnt = 0
            tes_already.append(te)

        if te in input_type.keys():
            date = input_type[te][3]
            cond_gen = input_type[te][1]

            if total_or_share == 'share':
                if cond_gen in ['co_all', 'obs_all']:
                    add = False
                else:
                    add = True
            elif total_or_share == 'total':
                if cond_gen in ['co_all', 'obs_all']:
                    add = True
                else:
                    add = False

            if add:
                if cond_gen[:2] == 'co':
                    c = 'co_all'
                elif cond_gen[:3] == 'obs':
                    c = 'obs_all'
                av = np.mean(avg_rpm[c, date])
                print av
                if norm == 'sub':
                    rpm_corr = rpm - av
                elif norm == 'div':
                    rpm_corr = rpm / av

                df_rpm['rpm'].append(rpm)
                df_rpm['rpm_corr'].append(rpm_corr)
                df_rpm['cond_gen'].append(input_type[te][1])
                df_rpm['te'].append(te)
                df_rpm['dec'].append(input_type[te][2])
                df_rpm['te_ord'].append(cnt)



    # for i, (ps, te) in enumerate(zip(hdf.root.meta_metrics[:]['all_percent_success'], hdf.root.meta_metrics[:]['task_entry'])):
    #   df_met['perc_succ'].append(np.mean(ps))
    #   df_met['cond_gen'].append(input_type[te][1])
    #   df_met['dec'].append(input_type[te][2])
    #   df_met['date'].append(input_type[te][3])

    #datfrm_met = pandas.DataFrame.from_dict(df_met)
    df['sa'] = np.array(df['t2t'])/np.array(df['pe'])
    
    datfrm = pandas.DataFrame.from_dict(df)
    datfrm_rpm = pandas.DataFrame.from_dict(df_rpm)
    return datfrm, datfrm_rpm

def plot_t2t(datfrm, combine_tasks=False, save=False, t2t_norm='sub'):
    # Plots by condition: 
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)
    met = 't2t_corr'

    # this is 'task_signal'
    if combine_tasks:
        conds = ['win_task', 'xtask']
    else:
        conds = np.unique(datfrm['cond_gen'])
    
    cond_nm = []
    x = {}
    dct = dict(cond=[], val=[], dat=[])

    for ii, i in enumerate(conds):
        # Get all of the same condition: 
        if combine_tasks:
            if i == 'win_task':
                ix0 = np.nonzero(datfrm['cond_gen']=='co_co_shar')[0]
                ix1 = np.nonzero(datfrm['cond_gen']=='obs_obs_shar')[0]
                ix = np.sort(np.hstack((ix0, ix1)))
            elif i == 'xtask':
                ix0 = np.nonzero(datfrm['cond_gen']=='co_obs_shar')[0]
                ix1 = np.nonzero(datfrm['cond_gen']=='obs_co_shar')[0]
                ix = np.sort(np.hstack((ix0, ix1)))                
        else:
            ix = np.nonzero(datfrm['cond_gen']==i)[0]
            
        #### NOTE THE -1 --> This makes this metric "faster than baseline" instead of "slower than baseline"
        if t2t_norm == 'sub':
            x[i] = -1*datfrm[met][ix] 
        
        elif t2t_norm == 'div':
            x[i] = datfrm[met][ix] 

        dct['val'].extend(datfrm[met][ix])
        dct['cond'].extend(np.zeros((len(ix)))+ii)
        dct['dat'].extend(datfrm['date'][ix])
        
        print len(ix), ii, i
        
        if i[:2] == 'ob':
            iii = ii + 1
        else:
            iii = ii
        
        cond_nm.append(i)
        if not combine_tasks:
            if ii == 1:
                cond_nm.append('')
        ax.bar(iii, np.mean(x[i]), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(iii, np.mean(x[i]), np.std(x[i])/np.sqrt(len(x[i])), marker='.', color='k')

    ax.set_xticks(np.arange(0, iii+1))
    ax.set_xticklabels(cond_nm, fontsize=14)
    ax.set_xlim([-.6, iii+.6])
    if t2t_norm == 'sub':
        ax.set_ylabel('Time to Target (secs faster than Baseline)', fontsize=14)
        if combine_tasks:
            print 'combo: ', scipy.stats.kruskal(x['win_task'], x['xtask'])
            print ''
            print ''
            print 'ttest: '
            print 'combo:', scipy.stats.ttest_ind(x['win_task'], x['xtask'])
            ax.plot([0, 1], [.4, .4], 'k-', linewidth=4.)
            ax.text(.5, .405, '***', fontsize=14)
            ax.set_ylim([0., .45])
            
        else:
            ax.plot([0, 1], [.275, .275], 'k-')
            ax.text(.5, .28, '*')

            ax.plot([3, 4], [.7, .7], 'k-')
            ax.text(3.5, .705, '*')
            ax.set_ylim([0., .75])
            print 'obs', scipy.stats.kruskal(x['obs_obs_shar'], x['obs_co_shar'])
            print 'co', scipy.stats.kruskal(x['co_obs_shar'], x['co_co_shar'])
            print ''
            print ''
            print 'ttest: '
            print 'obs', scipy.stats.ttest_ind(x['obs_obs_shar'], x['obs_co_shar'])
            print 'co', scipy.stats.ttest_ind(x['co_obs_shar'], x['co_co_shar'])

    elif t2t_norm == 'div':
        ax.set_ylabel('Percent of Baseline Time to Target', fontsize=14)
        if combine_tasks:
            print 'combo: ', scipy.stats.kruskal(x['win_task'], x['xtask'])
            print 'ttest: '
            print 'combo:', scipy.stats.ttest_ind(x['win_task'], x['xtask'])
            ax.set_ylim([0.8, 1.1])
            ax.plot([0, 1], [1.05, 1.05], 'k-', linewidth=4.)
            ax.text(.4, 1.075, '***', fontsize=14)
        else:
            print 'obs: ', scipy.stats.kruskal(x['obs_obs_shar'], x['obs_co_shar'])
            print 'co: ', scipy.stats.kruskal(x['co_obs_shar'], x['co_co_shar'])
            print 'ttest: '
            print 'obs', scipy.stats.ttest_ind(x['obs_obs_shar'], x['obs_co_shar'])
            print 'co', scipy.stats.ttest_ind(x['co_obs_shar'], x['co_co_shar'])

            ax.set_ylim([0.8, 1.1])
            ax.plot([0, 1], [1.05, 1.05], 'k-', linewidth=4.)
            ax.text(.5, 1.075, '*', fontsize=14)

            ax.plot([3, 4], [.95, .95], 'k-', linewidth=4.)
            ax.text(3.5, .975, '**', fontsize=14)


    plt.tight_layout()
    if save:
        f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/grom_co_obs_online_w_co_obs_shar_space_t2t_corr_norm'+str(t2t_norm)+'_combinextask'+str(combine_tasks)+'_wo_poor_data.svg', 
            transparent=True)

def plot_rpm_shar(datfrm_rpm, met = 'rpm'):
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)
    conds = np.unique(datfrm_rpm['cond_gen'])
    cond_nm = []
    #conds = ['co_all', 'co_obs_shar']
    x = {}

    for ii, i in enumerate(conds):
        ix1 = np.nonzero(datfrm_rpm['cond_gen']==i)[0]
        ix = ix1
        x[i] = datfrm_rpm[met][ix]
        print len(ix)
        if i[:2] == 'ob':
            iii = ii + 1
        else:
            iii = ii
        cond_nm.append(i)
        if ii == 1:
            cond_nm.append('')
        ax.bar(iii-.5, np.abs(np.mean(x[i])), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(iii, np.abs(np.mean(x[i])), np.std(x[i])/np.sqrt(len(x[i])), marker='.', color='k')

    ax.set_xticks(np.arange(0, iii+1))
    ax.set_xticklabels(cond_nm, rotation=45)
    ax.set_xlim([-.6, iii+.6])
    ax.set_ylabel('Rewards Per Min (more than Total)')

    ax.plot([0, 1], [9.5, 9.5], 'k-')
    ax.text(.5, 9.6, 'n.s')

    ax.plot([3, 4], [6, 6], 'k-')
    ax.text(3.5, 6.1, '**')
    ax.set_ylim([0., 10])
    print 'obs', scipy.stats.kruskal(x['obs_obs_shar'], x['obs_co_shar']), scipy.stats.ranksums(x['obs_obs_shar'], x['obs_co_shar'])
    print 'co', scipy.stats.kruskal(x['co_obs_shar'], x['co_co_shar']), scipy.stats.ranksums(x['co_obs_shar'], x['co_co_shar'])
    #f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/grom_co_obs_online_w_co_obs_shar_space_rpm_corr.svg', transparent=True)

def plot_any(datfrm, combine_tasks=False, norm='sub', save=False, met='pl_corr'):
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)
    if combine_tasks:
        conds = ['win_task', 'xtask']
    else:
        conds = np.unique(datfrm['cond_gen'])
    
    cond_nm = []
    x = {}
    for ii, i in enumerate(conds):
        if combine_tasks:
            if i == 'win_task':
                ix0 = np.nonzero(datfrm['cond_gen']=='co_co_shar')[0]
                ix1 = np.nonzero(datfrm['cond_gen']=='obs_obs_shar')[0]
                ix = np.sort(np.hstack((ix0, ix1)))
            elif i == 'xtask':
                ix0 = np.nonzero(datfrm['cond_gen']=='co_obs_shar')[0]
                ix1 = np.nonzero(datfrm['cond_gen']=='obs_co_shar')[0]
                ix = np.sort(np.hstack((ix0, ix1)))  
        else:
            ix = np.nonzero(datfrm['cond_gen']==i)[0]
        x[i] = datfrm[met][ix]
        print len(ix), ii, i
        if i[:2] == 'ob':
            iii = ii + 1
        else:
            iii = ii
        cond_nm.append(i)
        if not combine_tasks:
            if ii == 1:
                cond_nm.append('')
        ax.bar(iii, np.mean(x[i]), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(iii, np.mean(x[i]), np.std(x[i])/np.sqrt(len(x[i])), marker='.', color='k')

    ax.set_xticks(np.arange(0, iii+1))
    ax.set_xticklabels(cond_nm, fontsize=14)
    ax.set_xlim([-.6, iii+.6])

    if combine_tasks:
        print 'combo: ', scipy.stats.kruskal(x['win_task'], x['xtask'])
        print ''
        print ''
        print 'ttest: '
        print 'combo:', scipy.stats.ttest_ind(x['win_task'], x['xtask'])
        
    else:

        print 'obs', scipy.stats.kruskal(x['obs_obs_shar'], x['obs_co_shar'])
        print 'co', scipy.stats.kruskal(x['co_obs_shar'], x['co_co_shar'])

        print 'obs', scipy.stats.ttest_ind(x['obs_obs_shar'], x['obs_co_shar'])
        print 'co', scipy.stats.ttest_ind(x['co_obs_shar'], x['co_co_shar'])
    
    if save:
        if np.logical_and(met == 'avg_dt2_corr', norm=='sub'):
            if combine_tasks == False:
                plt.ylabel('Command Efficiency Towards Target \n more than baseline (cms)')
                plt.ylim([0., 0.017])

                plt.plot([0, 1], [.01, .01], 'k-', linewidth=4)
                plt.plot([3, 4], [.014, .014], 'k-', linewidth=4)
                plt.text(0.5, .0105, '*', fontsize=14)
                plt.text(3.5, .0145, '**', fontsize=14)
                
            else:
                plt.ylabel('Command Efficiency Towards Target \n more than baseline (cms)')
                plt.ylim([0., 0.013])
                plt.plot([0, 1], [.011, .011], 'k-', linewidth=4)
                plt.text(0.5, .0115, '***', fontsize=14)
        plt.tight_layout()
        f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/grom_co_obs_online_w_co_obs_shar_space_'+met+'_norm'+norm+'_combine'+str(combine_tasks)+'.svg', transparent=True)

def plot_co_vs_obs_dec(datfrm_rpm, cnt_lte=2):
    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)
    met = 'rpm'
    conds = ['co_all', 'obs_all']
    dec_conds = ['co', 'obs']
    cond_nm = []

    x = {}
    for ii, i in enumerate(conds):
        ix1 = np.nonzero(datfrm_rpm['cond_gen']==i)[0]
        for jj, j in enumerate(dec_conds):
            ix2 = np.nonzero(datfrm_rpm['dec'][ix1]==j)[0]
            ix3 = np.nonzero(datfrm_rpm['te_ord'][ix1[ix2]] <= cnt_lte)[0]
            ix = ix1[ix2[ix3]]    
            print i, j, np.mean(datfrm_rpm[met][ix])
            x[i, j] = datfrm_rpm[met][ix]
            #ax.bar((ii*2)+jj-.5+ii, np.mean(x[i, j]), width=1.,edgecolor='k', linewidth=4., color='w')
            #ax.errorbar((ii*2)+jj+ii, np.mean(x[i, j]), np.std(x[i, j])/np.sqrt(len(x[i, j])), marker='.', color='k')
            #cond_nm.append(i+'_'+j)
        if ii == 0:
            cond_nm.append('')

    same = np.hstack((x['co_all', 'co'], x['obs_all','obs']))
    diff = np.hstack((x['co_all', 'obs'], x['obs_all', 'co']))
    ax.bar(0, np.mean(same), width=1.,edgecolor='k', linewidth=4., color='w')
    ax.errorbar(0, np.mean(same), np.std(same)/np.sqrt(len(same)), marker='.', color='k')
    ax.bar(1, np.mean(diff), width=1.,edgecolor='k', linewidth=4., color='w')
    ax.errorbar(1, np.mean(diff), np.std(diff)/np.sqrt(len(diff)), marker='.', color='k')

    ax.set_xlim([-1, 2]) #(ii*2)+jj+.6+ii])
    ax.set_ylabel('Rewards Per Min')
    ax.set_xticks([0., 1.])
    ax.set_xticklabels(['Same Decoder\n as Task', 'Opposite Decoder\n as Task'])
    ax.plot([0., 1.], [23, 23], 'k-')
    ax.text(.5, 23.5, '*')

    # ax.plot([3, 4], [19., 19.], 'k-')
    # ax.text(3.5, 19.1, '*')
    ax.set_ylim([0., 24])
    print 'co diffs', scipy.stats.kruskal(x['co_all', 'co'], x['co_all', 'obs']), scipy.stats.ranksums(x['co_all', 'co'], x['co_all', 'obs'])
    print 'obs diffs', scipy.stats.kruskal(x['obs_all', 'co'], x['obs_all', 'obs']), scipy.stats.ranksums(x['obs_all', 'co'], x['obs_all', 'obs'])
    print 'same vs. diff', scipy.stats.kruskal(np.hstack((x['co_all', 'co'], x['obs_all','obs'])), np.hstack((x['co_all', 'obs'], x['obs_all', 'co'])))
    print 'same n: ', len(np.hstack((x['co_all', 'co'], x['obs_all','obs']))), ' diff n: ', len(np.hstack((x['co_all', 'obs'], x['obs_all', 'co'])))
    #f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/grom_co_obs_online_w_co_obs_dec_rpm_lumped.svg', transparent=True)

def get_te_decoder_mismatch_w_C(input_type):
    te_list = [k for k in input_type.keys() if input_type[k][1] in ['obs_all', 'co_all']]
    d = {}

    for te in te_list:
        # Train a KF w/ task entry: 
        C_task, C_dec = train_KFdecoder_from_teid(te)
        d[te, 'C_task'] = C_task
        d[te, 'C_dec'] = C_dec
    return d

def train_KFdecoder_from_teid(te):
    import pickle
    from riglib.bmi import train
    #decoder = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/grom20160307_02_RMLC03071555.pkl'))
    binlen = 0.1

    # From new TE
    from db import dbfunctions as dbfn
    te_id = dbfn.TaskEntry(te)
    files = dict(plexon=te_id.plx_filename, hdf = te_id.hdf_filename)
    
    # Use from old decoder:
    decoder = te_id.decoder
    extractor_cls = decoder.extractor_cls
    extractor_kwargs = decoder.extractor_kwargs
    extractor_kwargs['discard_zero_units'] = False
    kin_extractor = train.get_plant_pos_vel
    ssm = decoder.ssm
    update_rate = decoder.binlen
    units = decoder.units
    tslice = (0., te_id.length)

    ## get kinematic data
    kin_source = 'task'
    tmask, rows = train._get_tmask(files, tslice, sys_name=kin_source)
    kin = kin_extractor(files, binlen, tmask, pos_key='cursor', vel_key=None)

    ## get neural features
    neural_features, units, extractor_kwargs = train.get_neural_features(files, binlen, extractor_cls.extract_from_file, extractor_kwargs, tslice=tslice, units=units, source=kin_source)
    kin = kin[1:].T
    neural_features = neural_features[:-1].T
    decoder2 = train.train_KFDecoder_abstract(ssm, kin, neural_features, units, update_rate, tslice=tslice)

    C_task = decoder2.filt.C
    C_dec = te_id.decoder.filt.C
    return C_task, C_dec

def plot_decoder_mismatch(input_type):
    try:
        d = pickle.load(open('/Volumes/TimeMachineBackups/grom2016/grom2017_co_obs_task_vs_dec_C_matrices.pkl'))
    except:
        d = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2017_co_obs_task_vs_dec_C_matrices.pkl'))
    te_list = [k for k in input_type.keys() if input_type[k][1] in ['obs_all', 'co_all']]
    print te_list, len(te_list)
    ang_mismatch = dict(co_co_dec = [], co_obs_dec=[], ob_co_dec=[], ob_obs_dec=[])
    
    for it, te in enumerate(te_list):
        C_task = d[te, 'C_task']
        C_dec = d[te, 'C_dec']
        key = input_type[te][0][:2]+'_'+input_type[te][2]+'_dec'
        nunits = C_task.shape[0]
        try:
            assert nunits == C_dec.shape[0]
            p = True
        except:
            p = False
            print 'non-matching: ', C_dec.shape[0], nunits, te

        if p:
            tmp = []
            for i in range(nunits):
                ctsk = np.squeeze(np.array(C_task[i, [3, 5]]))
                cdec = np.squeeze(np.array(C_dec[i, [3, 5]]))
                dang = ctsk-cdec
                #dtheta = np.abs(math.atan2(dang[1], dang[0]))
                tmp.append(dang)
            mndang = np.mean(np.vstack((tmp)), axis=0)
            ang_mismatch[key].append(np.abs(math.atan2(mndang[0], mndang[1])))

    f, ax = plt.subplots()
    f.set_figheight(5)
    f.set_figwidth(6)
    nms = []
    conds = ['co_all', 'obs_all']
    dec_conds = ['co', 'obs']

    same_diff = [['co_co_dec', 'ob_obs_dec'], ['co_obs_dec', 'ob_co_dec']]

    #for ic, c in enumerate(conds):
        #for ij, j in enumerate(dec_conds):
    for i, sd in enumerate(same_diff):
        x = []
        for s in sd:
            x.append(ang_mismatch[s])
        x = np.hstack((x))
        ax.bar(i, np.mean(x), width=1.,edgecolor='k', linewidth=4., color='w')
        ax.errorbar(i, np.mean(x), np.std(x)/np.sqrt(len(x)), marker='.', color='k')
            #key = c[:2]+'_'+j+'_dec'
            #ax.bar((3*ic)+ij-.5, np.mean(ang_mismatch[key]), width=1.,edgecolor='k', linewidth=4., color='w')
            #ax.errorbar((3*ic)+ij, np.mean(ang_mismatch[key]), np.std(ang_mismatch[key])/np.sqrt(len(ang_mismatch[key])), marker='.', color='k')
            #nms.append(key)
        #if ic == 0:
        #    nms.append('')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Same Decoder\n as Task', 'Opposite Decoder\n as Task'])
    ax.set_xlim([-1, 2])
    ax.set_ylabel('Mean Angular Mismatch')

    ax.plot([0., 1.], [2.7, 2.7], 'k-')
    ax.text(.5, 2.75, 'p=0.061')

    #ax.plot([3, 4], [2.9, 2.9], 'k-')
    #ax.text(3.5, 3.0, '**')
    ax.set_ylim([0., 3.1])
    print 'co diffs', scipy.stats.kruskal(ang_mismatch['co_co_dec'], ang_mismatch['co_obs_dec']), scipy.stats.ranksums(ang_mismatch['co_co_dec'], ang_mismatch['co_obs_dec'])
    print 'obs diffs', scipy.stats.kruskal(ang_mismatch['ob_co_dec'], ang_mismatch['ob_obs_dec']), scipy.stats.ranksums(ang_mismatch['ob_co_dec'], ang_mismatch['ob_obs_dec'])
    print 'same vs. diff', scipy.stats.kruskal(np.hstack((ang_mismatch['co_co_dec'], ang_mismatch['ob_obs_dec'])), np.hstack((ang_mismatch['co_obs_dec'], ang_mismatch['ob_co_dec'])))
    print 'same n: ', len(np.hstack((ang_mismatch['co_co_dec'], ang_mismatch['ob_obs_dec']))), ' diff n: ', len(np.hstack((ang_mismatch['co_obs_dec'], ang_mismatch['ob_co_dec'])))
    f.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/grom_co_obs_online_w_co_obs_dec_task_dec_mismatch_lumped.svg', transparent=True)




# Plot meta condition:
# f, ax = plt.subplots()
# met = 'perc_succ'
# conds = np.unique(datfrm_met['cond_gen'])
# x = {}
# for ii, i in enumerate(conds):
#   ix = np.nonzero(datfrm_met['cond_gen']==i)[0]
#   x[i] = datfrm_met[met][ix]
#   ax.bar(ii+.4, np.mean(x[i]))
# ax.set_xticks(range(ii+1))
# ax.set_xticklabels(conds)



# # Plots by condition: 
# f, ax = plt.subplots()
# met = 'rpm'
# #conds = ['co_all', 'co_co_shar']
# conds = ['co_all', 'co_obs_shar']
# #dates = ['3-16', '3-22', '3-23', '3-24']
# dates = ['3-15', '3-16', '3-17', '3-19']
# xoff = 0
# for ii, i in enumerate(conds):
#     ix1 = np.nonzero(datfrm['cond_gen'] == i)[0]
#     ix2 = np.array([i for i, j in enumerate(datfrm['date'][ix1]) if j in dates])
#     ix = ix1[ix2]
#     x[i] = datfrm[met][ix]
#     ax.bar(ii, np.mean(x[i]))
    


# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ix = np.array([i for i, j in enumerate(datfrm['cond_gen']) if j[:2] == 'co'])
# ix2 = np.array([i for i, j in enumerate(datfrm['cond_gen']) if j[:2] == 'ob'])


# formula = 'sa ~ date + cond_gen'
# model = ols(formula, datfrm.ix[ix]).fit()
# aov_table = anova_lm(model, typ=2)
# print(pairwise_tukeyhsd(datfrm['t2t'].ix[ix], datfrm['cond_gen'].ix[ix], alpha=0.05))


# formula = 'sa ~ date + cond_gen'
# model = ols(formula, datfrm.ix[ix2]).fit()
# aov_table = anova_lm(model, typ=2)
# print(pairwise_tukeyhsd(datfrm['t2t'].ix[ix2], datfrm['cond_gen'].ix[ix2], alpha=0.05))

# # # Using Obstacle shared to drive CO BMI resetting: 
# # x = []
# # x.append([[4499, 4503], ['all', 'mn_sc_shar']]) # obs decoder
# # x.append([[4510, 4511, 4513], ['all', 'mn_sc_shar', 'sc_shar']]) #obs decoder
# # x.append([[4523, 4524, 4525], ['all', 'mn_sc_shar', 'all']]) #obs decoder
# # x.append([[4535, 4536, 4537], ['mn_shar', 'all', 'mn_shar']]) #obs decoder
# # x.append([[4553, 4554, 4558], ['all', 'mn_sc_shar', 'all']]) #obs decoder

# # te_list = []
# # for i, (te, nm) in enumerate(x):
# #   for t in te:
# #       te_list.append(t)
# # new_hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_use_obs_shar_on_CO_task.hdf')
# # basic_hdf_analysis.process_targets(te_list, new_hdf_name)


# # # Using Obstacle shared to drive Obs. BMI resetting: 
# # x = []
# # x.append([[4497, 4498, 4504, 4505], ['all','all', 'all', 'mn_sc_shar']]) #obs decoder
# # x.append([[4509, 4514, 4515], ['all', 'all', 'mn_sc_shar']]) # 151 units in everything #obs decoder
# # x.append([[4520, 4521, 4520, 4526, 4528], ['all', 'mn_sc_shar', 'all', 'all', 'mn_shar']]) #obs decoder
# # x.append([[4532, 4533, 4534, 4538], ['all', 'all', 'mn_shar', 'mn_shar']]) #obs decoder
# # x.append([[4549, 4550, 4552, 4559, 4560], ['all', 'mn_sc_shar', 'mn_sc_shar', 'mn_sc_shar', 'all']]) #obs decoder
# # te_list = []
# # for i, (te, nm) in enumerate(x):
# #   for t in te:
# #       te_list.append(t)
# # new_hdf_name = os.path.expandvars('$FA_GROM_DATA/grom2016_use_obs_shar_on_obs_task.hdf')
# # basic_hdf_analysis.process_targets(te_list, new_hdf_name)