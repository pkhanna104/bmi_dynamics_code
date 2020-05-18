import numpy as np; 
import math 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

import matplotlib.pyplot as plt

def get_cov_diffs(ix_co, ix_ob, spks, diffs, method = 1):
    cov1 = np.cov(10*spks[ix_co, :].T); 
    cov2 = np.cov(10*spks[ix_ob, :].T); 

    ### Look at actual values of cov matrix
    if method == 1: 
        ix_upper = np.triu_indices(cov1.shape[0])
        diffs.append(cov1[ix_upper] - cov2[ix_upper])

    ### Do subpsace overlap -- both ways; 
    elif method == 2: 
        ov1, _ = subspace_overlap.get_overlap(None, None, first_UUT=cov1, second_UUT=cov2, main=False)# main_thresh = 0.9)
        ov2, _ = subspace_overlap.get_overlap(None, None, first_UUT=cov2, second_UUT=cov1, main=False)#, main_thresh = 0.9)
        diffs.append(ov1)
        diffs.append(ov2)

    return diffs

def commands2bins(commands, mag_boundaries, animal, day_ix, vel_ix = [3, 5], ndiv=8):
    mags = mag_boundaries[animal, day_ix]
    rads = np.linspace(0., 2*np.pi, ndiv+1) + np.pi/8
    command_bins = []
    for com in commands: 
        ### For each trial: 
        T = com.shape[0]
        vel = com[:, vel_ix]
        mag = np.linalg.norm(vel, axis=1)
        ang = []
        for t in range(T):
            ang.append(math.atan2(vel[t, 1], vel[t, 0]))
        ang = np.hstack((ang))

        ### Re-map from [-pi, pi] to [0, 2*pi]
        ang[ang < 0] += 2*np.pi

        ### Digitize: 
        ### will bin in b/w [m1, m2, m3] where index of 0 --> <m1, 1 --> [m1, m2]
        mag_bins = np.digitize(mag, mags)

        ###
        ang_bins = np.digitize(ang, rads)
        ang_bins[ang_bins == 8] = 0; 

        command_bins.append(np.hstack((mag_bins[:, np.newaxis], ang_bins[:, np.newaxis])))
    return command_bins

def get_angles():
    rang = np.linspace(-2*np.pi, 2*np.pi, 200)
    ang = np.linspace(-2*np.pi, 2*np.pi, 200)
    ang[ang < 0] += 2*np.pi
    ang_bins = np.digitize(ang, ads)
    ang_bins[ang_bins == 8] = 0; 
    plt.plot(rang, ang)
    plt.plot(rang, ang_bins)

#### Linear mixed effect modeling: 
def run_LME(Days, Grp, Metric, bar_plot = False, xlabels = None, title = ''):
    data = pd.DataFrame(dict(Days=Days, Grp=Grp, Metric=Metric))
    md = smf.mixedlm("Metric ~ Grp", data, groups=data["Days"])
    mdf = md.fit()
    pv = mdf.pvalues['Grp']
    slp = mdf.params['Grp']

    if bar_plot:
        f, ax = plt.subplots(figsize = (4, 4))
        ix = np.unique(Grp)
        ymax = 0; 

        for i in ix:
            ixx = np.nonzero(Grp == i)[0]
            ax.bar(i, np.mean(Metric[ixx]))
            ax.errorbar(i, np.std(Metric[ixx])/np.sqrt(len(ixx)), marker='|', color='k')
            ymax = np.max(np.array([ymax, np.mean(Metric[ixx])]))

        ixd = np.unique(Days)
        for i_d in ixd:
            tmp = []
            for i in ix: 
                ixx = np.nonzero(np.logical_and(Days == i_d, Grp == i))[0]
                tmp.append(np.mean(Metric[ixx]))
            ax.plot(ix, tmp, 'k-')

        ax.plot([ix[0], ix[-1]], 1.2*np.array([ymax, ymax]), 'k-')

        if pv < 0.001:
            string = '***'
        elif pv < 0.01:
            string = '**'
        elif pv < 0.05:
            string = '*'
        else:
            string = 'n.s.'
        ax.text(np.mean(np.array([ix[0], ix[-1]])), 1.3*ymax, string, color='k')
        ax.set_ylim([0., 1.4*ymax])
        if xlabels is None: 
            pass
        else:
            ax.set_xticks(np.unique(Grp))
            ax.set_xticklabels(xlabels, rotation = 45)
        ax.set_title(title)
        f.tight_layout()



    return pv, slp

def get_R2(y_true, y_pred, pop = True, ignore_nans = False):
    assert(y_true.shape == y_pred.shape)

    if np.logical_and(len(y_true.shape) == 1, len(y_pred.shape) == 1):
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]
    if ignore_nans:
        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.nansum((y_true - y_pred)**2, axis=0)
        SST_i = np.nansum((y_true - np.nanmean(y_true, axis=0)[np.newaxis, :])**2, axis=0)

        if pop:
            return 1 - np.nansum(SSR_i)/np.nansum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)
    else: 

        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.sum((y_true - y_pred)**2, axis=0)
        SST_i = np.sum((y_true - np.mean(y_true, axis=0)[np.newaxis, :])**2, axis=0)

        if pop:
            return 1 - np.sum(SSR_i)/np.sum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)
### Plotting ###
def draw_plot(xax, data, edge_color, fill_color, ax, width = .5):
    bp = ax.boxplot(data, patch_artist=True, positions = [xax], widths=[width])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 

def plot_mean_and_sem(x , array, ax, color='b', array_axis=1,label='0',
    log_y=False, make_min_zero=[False,False]):
    
    mean = array.mean(axis=array_axis)
    sem_plus = mean + scipy.stats.sem(array, axis=array_axis)
    sem_minus = mean - scipy.stats.sem(array, axis=array_axis)
    
    if make_min_zero[0] is not False:
        bi, bv = get_in_range(x,make_min_zero[1])
        add = np.min(mean[bi])
    else:
        add = 0

    ax.fill_between(x, sem_plus-add, sem_minus-add, color=color, alpha=0.5)
    x = ax.plot(x,mean-add, '-',color=color,label=label)
    if log_y:
        ax.set_yscale('log')
    return x, ax