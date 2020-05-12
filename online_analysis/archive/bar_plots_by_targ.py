import tables
import seaborn
import basic_hdf_analysis as bha
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)

#Seaborn defaults
seaborn.set(font='Arial',context='talk',font_scale=1.5,style='white')

def bar_plots_from_trial_metrics(hdf_filename, hdf_column, ylabel=None, ylim=None,
    stats=True, **kwargs):
    trial_mets = tables.openFile(hdf_filename)
    return plot_by_targ_and_input_type(hdf_column, trial_mets, ylabel=ylabel, ylim=ylim, stats=stats, **kwargs)


# Plot input type by trial type by target: 
def plot_by_targ_and_input_type(metric_wanted, met_hdf, ylabel=None, ylim=None, stats=True, 
    input_type=None, collapse_targ=False, subset_tes=None, subset_cond=None, return_ax = False, 
    ax=None, x_axis_offs=0, bar_plot_instead=False,input_type2=None, metric_scale=1):


    #Init figure
    if collapse_targ:
        if ax is None:
            f, ax = plt.subplots()
    else:
        #fig = plt.figure(figsize=(10,10))
        f, ax = plt.subplots(nrows=3, ncols=3)
    
    if subset_cond is None:
        add_subset = ''
    else:
        add_subset = ' & '+subset_cond

    targs = np.unique(met_hdf.root.trial_metrics[:]['target_index'])

    if subset_tes is None:
        subset_tes = np.unique(met_hdf.root.trial_metrics[:]['task_entry'])
        ix = np.arange(len(met_hdf.root.trial_metrics))
    else:
        ix = np.array([i for i, j in enumerate(met_hdf.root.trial_metrics[:]['task_entry']) if j in subset_tes])

    if input_type is None:
        inputs = np.unique(met_hdf.root.trial_metrics[ix]['input_type'])
    else:
        tmp_inp = np.array([input_type[te] for te in subset_tes])
        inputs = np.unique(tmp_inp)

    tr_table = met_hdf.root.trial_metrics
    master_times = []
    #Iterate through target num (target_index in hdf file)
    for it in targs:
        
        #Init empty arryays and names:
        times = []
        nms = []
        
        #Iterate through 'fa_inputs':
        for inp in inputs:
            #Set pytable query
            if input_type is None:
                cond = '(target_index == it) & (input_type == inp)'+add_subset
            
            else:
                inp_times = []
                #Get TEs w/ correct input: 
                te_arr = np.array([te for te in input_type.keys() if input_type[te] == inp])
                cond = []
                for te in te_arr:
                    cond.append('(target_index == it) & (task_entry == te)'+add_subset)
            
            #Add desired metric of trials that are of correct input type and target index
            if type(cond) == str:
                if metric_wanted == 'avg_path_error':
                    times.append([x['path_error']/x['time2targ'] for x in tr_table.where(cond)])
                else:
                    times.append([x[metric_wanted]*metric_scale for x in tr_table.where(cond)])

            else:
                for c in cond:
                    inp_times.extend([x[metric_wanted]for x in tr_table.where(c)])
                times.append(inp_times)

            nms.append(inp)


        if collapse_targ:
            if len(master_times) == 0:
                master_times = times
            else:
                for i, t in enumerate(times):
                    master_times[i].extend(times[i])
        
        else:
            #Map target index to correct subplot location
            axi = ax[bha.targ_ix_to_3x3_subplot(it)]
            
            #Boxplot
            if bar_plot_instead:
                tz = np.array([np.mean(t) for t in times])
                stz = np.array([np.std(t)/np.sqrt(len(times)) for t in times])
                z = seaborn.barplot(np.arange(len(times))+x_axis_offs, tz) #facecolor=(1, 1, 1, 0))
                z.errorbar(np.arange(len(times))+x_axis_offs, stz, ecolor=(0, 0, 0, 1) )
                z.set_xticklabels(nms)
            
            else:
                z = seaborn.boxplot(times, names=nms, ax=axi, fliersize=0, whis=0., positions=np.arange(len(times))+x_axis_offs)
            plt.setp(z.get_xticklabels(), rotation=45)

            #Plot stats:
            if stats:
                print 'stats for targ ix: ', it, len(times), len(nms)
                axi, dy = add_stats(axi, times, nms, ylim, x_axis_offs=x_axis_offs)

            if ylim is not None:
                axi.set_ylim([ylim[0], ylim[1]+6*dy])

    if collapse_targ:

        if bar_plot_instead:
            tz = np.array([np.mean(t) for t in master_times])
            stz = np.array([np.std(t)/np.sqrt(len(master_times)) for t in master_times])
            z = seaborn.barplot(np.arange(len(times))+x_axis_offs, tz, color=(.75, .75, .75, 1))
            z.set_xticklabels(nms)
            z.errorbar(np.arange(len(times))+x_axis_offs, tz, stz,fmt='.', color='k')
            print 'stz, ', stz, type(stz), np.arange(len(times))+x_axis_offs,
            z.set_xticklabels(nms)

        else:
            z = seaborn.boxplot(master_times, names=nms, ax=ax, fliersize=0, whis=0., positions=np.arange(len(times))+x_axis_offs)
        plt.setp(z.get_xticklabels(), rotation=45)
        if ylabel is not None:
                z.set_ylabel(ylabel)
    
        if stats:
            print 'stats for master'
            axi, dy = add_stats(ax, master_times, nms, ylim, x_axis_offs=x_axis_offs)

        if ylim is not None:
            ax.set_ylim([ylim[0], ylim[1]+6*dy])
        #ax.set_title(metric_wanted)
    
    else:       
        #ax[0, 1].set_title(metric_wanted)
    
        if ylabel is not None:
            for i in range(3):
                ax[i, 0].set_ylabel(ylabel)
                
    #Magic
    #plt.tight_layout()
    if return_ax:
        return ax

def bar_plot_meta_mets(metric_wanted, met_hdf, input_type, ylabel=None, ylim=None,subset_tes=None,
    return_ax = False, ax = None, x_axis_offs=0):
    
    if ax is None:
        f, ax = plt.subplots()
        print ax, 'ax:'
    else:
        print 'using ax'
    meta_table = met_hdf.root.meta_metrics
    if subset_tes is None:
        task_entries = np.unique(meta_table[:]['task_entry'])
    else:
        task_entries = subset_tes

    inputs = np.vstack((input_type.items()))
    inputs = np.unique(inputs[:,1])

    master_times = []

    input_ = []
    met_ = []
    for inp in inputs:
        input_.append(inp)

        te_array = np.array([te for te in input_type.keys() if input_type[te] == inp])
        sub_met_ = []

        for te in te_array:
            cond = '(task_entry == te)'
            sub_met_.extend(x[metric_wanted] for x in meta_table.where(cond))

        met_.append(sub_met_)
    
    z= seaborn.boxplot(met_, names=input_, ax=ax, fliersize=0, whis=0., positions=np.arange(len(input_))+x_axis_offs)
    plt.setp(z.get_xticklabels(), rotation=45)
    #axi, dy = add_stats(ax, met_, input_, ylim)
    ax.set_title(metric_wanted)
    plt.tight_layout()


def add_stats(ax, times, names, ylim, x_axis_offs=None):
    to_plot= []

    #Find y coordinate to plot at: 
    if ylim is None:
        y_ = np.max(np.max(np.array(times)))
        dy = .05*(y_ - np.min(np.min(np.array(times))))

    else:
        y_ = ylim[1]- (.025*(ylim[1]-ylim[0]))
        dy = .05*(ylim[1]-ylim[0])

    #Test all 3 distributions for normality: 
    abnorm_cnt = 0
    for tt, tm_arr in enumerate(times):
        print names[tt], len(tm_arr)
        zscore_tm_arr = ( np.array(tm_arr) - np.mean(np.array(tm_arr)) )/np.std(np.array(tm_arr))
        x, p = scipy.stats.kstest(zscore_tm_arr,'norm')
        if p < 0.05:
            #Not normal (two-sided test)
            abnorm_cnt += 1
            #Print 
            print 'non-normal: ', names[tt]

    #If all are normal, use ANOVA + Tukey's HSD
    if abnorm_cnt == 0:
        #All normal distributions:
        F, p = scipy.stats.f_oneway(*times)
        if p < 0.05:
            #Passed ANOVA, continue with Tukey's LSD test
            print 'passed anova!'
            for i, tm_i in enumerate(times):
                for j, tm_j in enumerate(times[i+1:]):
                    X = np.hstack(( np.array(tm_i), np.array(tm_j) ))
                    Y = np.hstack(( np.zeros((len(tm_i))), np.ones((len(tm_j))) ))
                    res2 = pairwise_tukeyhsd(X, Y)

                    if res2.reject:
                        to_plot.append([i, j+i+1, res2.reject, 'norm'])
                        print 'sig! ', [i, j+i+1, res2.reject, 'norm'], names[i], names[j+i+1]

    #If any is not normal, use KW + MW:
    else:
        #Use KW test: 
        H, p = scipy.stats.mstats.kruskalwallis(*times)
        if p < 0.05:
            #Passed group test, continue w/ Mann Whitney test:
            for i, tm_i in enumerate(times):
                for j, tm_j in enumerate(times[i+1:]):
                    u, p_onesided = scipy.stats.mannwhitneyu(tm_i, tm_j)
                    if (p_onesided*2.) < 0.05:
                        to_plot.append([i, j+i+1, p_onesided*2., 'nonnorm'])
                        print 'sig! ', [i, j+i+1, p_onesided*2., 'nonnorm']
                    else:
                        print 'non-sig: ', p_onesided*2.
        else:
            print 'non-sig KW: ', p

    if len(to_plot) > 0:
        for plt_line in to_plot:
            # ax, x1, x2, y, dy, pvalue
            ax = plot_sig_line(ax, plt_line[0], plt_line[1], y_, dy, plt_line[2], plt_line[3], x_axis_offs=x_axis_offs)
    return ax, dy

def plot_sig_line(ax, x1, x2, y, dy, pval, stat_method, x_axis_offs=0):
    dx = 0.1
    nudge_x = 0.5
    dy_3 = dy/3.

    if np.logical_and(pval>0.01, pval<=0.05):
        psymb = '*'
    elif np.logical_and(pval>0.001, pval<=0.01):
        psymb = '**'
    elif pval<=0.001:
        psymb = '***'
    else:
        psymb = '*'

    if stat_method == 'norm':
        pcol = 'k'
    elif stat_method == 'nonnorm':
        pcol = 'k'

    # Plot line first: 
    if x2 - x1 == 1:
        ax.plot([x1+1+dx+x_axis_offs-1, x2+1-dx+x_axis_offs-1], [y+dy, y+dy], '-', color=pcol)
        ax.text(((x1+x2+1+2*x_axis_offs-2)/2.) +nudge_x, y+dy-(dy_3), psymb, color=pcol, fontdict=dict(size=20), 
            horizontalalignment='center')
    else:
        ax.plot([x1+1+dx+x_axis_offs-1, x2+1-dx+x_axis_offs-1], [y+dy+(3*dy), y+dy+(3*dy)], '-', color=pcol)
        ax.text(((x1+x2+1+2*x_axis_offs-2)/2.) +nudge_x, y+dy+(3*dy)-(dy_3), psymb, color=pcol, fontdict=dict(size=20), 
            horizontalalignment='center')
    return ax



