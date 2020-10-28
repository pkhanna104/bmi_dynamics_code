import numpy as np; 
import math 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

import scipy.io as sio
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import pickle

from resim_ppf import file_key

import analysis_config

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

#### discretize commands #####
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

### reverse bin numbers based on trials ####
def bin_num_relative_to_end(bin_num): 
    '''
    method to index bin relative to the end of the trial 
    '''
    bin_cnt = []
    for b in bin_num:
        bin_cnt.append(b[::-1])
    bin_cnt = np.hstack((bin_cnt))
    bin_num = np.hstack((bin_num))
    
    ix, _ = scipy.signal.find_peaks(bin_num)
    assert(np.all(bin_cnt[ix] == 0))
    
    ix, _ = scipy.signal.find_peaks(bin_cnt)
    assert(np.all(bin_num[ix] == 0))
    
    return np.hstack((bin_cnt))

### target fcns #######
def add_targ_locs(data_temp, animal):

    targ_ix = data_temp['trg']
    tsk_ix = data_temp['tsk']
    targ_pos = get_targ_pos(animal, targ_ix.astype(int), tsk_ix.astype(int))

    data_temp['trgx'] = targ_pos[:, 0]
    data_temp['trgy'] = targ_pos[:, 1]

    if animal == 'grom': 
        data_temp['centx'] = np.zeros((len(targ_ix)))
        data_temp['centy'] = np.zeros((len(targ_ix)))
        
        data_temp['obsx'] = np.zeros((len(targ_ix)))
        data_temp['obsy'] = np.zeros((len(targ_ix)))

        #### Divide target by 2. to get obstacle locations 
        obs_ix = np.nonzero(tsk_ix == 1)[0]
        data_temp['obsx'][obs_ix] = data_temp['trgx'][obs_ix] / 2.
        data_temp['obsy'][obs_ix] = data_temp['trgy'][obs_ix] / 2.

    elif animal == 'jeev':
        centerPos = np.array([ 0.0377292,  0.1383867])
        data_temp['centx'] = 20*(np.zeros((len(targ_ix))) + centerPos[0])
        data_temp['centy'] = 20*(np.zeros((len(targ_ix))) + centerPos[1])
        
        data_temp['obsx'] = np.zeros((len(targ_ix)))
        data_temp['obsy'] = np.zeros((len(targ_ix)))
        
        co_ix = np.nonzero(tsk_ix == 0)[0]
        data_temp['trgx'][co_ix] = 20*data_temp['trgx'][co_ix]
        data_temp['trgy'][co_ix] = 20*data_temp['trgy'][co_ix]
        

        obs_ix = np.nonzero(tsk_ix == 1)[0]
        obs_tg = get_obs_pos_jeev(targ_ix[obs_ix].astype(int))

        data_temp['obsx'][obs_ix] = obs_tg[:, 0]
        data_temp['obsy'][obs_ix] = obs_tg[:, 1]

        ### Convert obs stuff to same coordinates as pos / vel; 
        for _, (ik, key) in enumerate(zip([0, 1, 0, 1], ['trgx', 'trgy', 'obsx', 'obsy'])):
            tg = np.squeeze(np.array(data_temp[key][obs_ix] - np.array([0., 2.5])[ik]))

            ### Convert cm --> m 
            tg = tg / 100.

            ### Add back the other center; 
            tg = tg + centerPos[ik]

            ### Substitute back in; 
            data_temp[key][obs_ix] = 20*tg.copy()

    return data_temp

def get_targ_pos(animal, targ_ix, task_ix):
    if animal == 'grom':
        dats = sio.loadmat(analysis_config.config['grom_pref'] + 'unique_targ.mat')
        unique_targ = dats['unique_targ']
        targ_pos = np.zeros((len(targ_ix), 2))
        for i_t, tg_ix in enumerate(targ_ix):
            targ_pos[i_t, :] = unique_targ[tg_ix, :]

    elif animal == 'jeev':
        targ_pos = np.zeros((len(targ_ix), 2))
        ix0 = np.nonzero(task_ix == 0)[0]
        ix1 = np.nonzero(task_ix == 1)[0]

        targ_pos[ix0] = file_key.targ_ix_to_loc(targ_ix[ix0])

        ### Get obstacle target location; 
        targ_pos[ix1] = file_key.targ_ix_to_loc_obs(targ_ix[ix1])

    return targ_pos

def get_obs_pos_jeev(targ_ix):
    jdat = sio.loadmat(analysis_config.config['BMI_DYN'] + 'resim_ppf/jeev_obs_positions_from_amy.mat')
    obs_pos = []
    for it, tg in enumerate(targ_ix):
        int_targ_num = jdat['trialList'][tg, 1] - 1 #Python indexing
        obs_pos.append(jdat['targPos'][int_targ_num, :])
    return np.vstack((obs_pos))

def get_angles():
    rang = np.linspace(-2*np.pi, 2*np.pi, 200)
    ang = np.linspace(-2*np.pi, 2*np.pi, 200)
    ang[ang < 0] += 2*np.pi
    ang_bins = np.digitize(ang, ads)
    ang_bins[ang_bins == 8] = 0; 
    plt.plot(rang, ang)
    plt.plot(rang, ang_bins)

def hstack_keys(d):
    for k in d.keys():
        d[k] = np.hstack((d[k]))
    return d
  
#### Get the CW/CCW target traj ####
def get_target_cw_ccw(animal, day_ix, cursor_pos, targIx): 
    """
    Method to get target index depending on whether 
    trajectory goes around the obstacle CW or CCW: 
    
    Args:
        animal (str): Description
        cursor_pos (np.array T x 2): trial time x (pos_x, pos_y)
        targIx (int): target index
    
    Returns:
        float: targIx + 0. if CW or targIx + 0.1 if CCW
    """
    if animal == 'jeev':
        if targIx in [1, 3, 4, 5, 8, 9]: 
            cw_ccw = analysis_config.jeev_cw_ccw_dict[targIx]/10.
        else:
            #### Test obstacles ####
            x = sio.loadmat('resim_ppf/jeev_obs_positions_from_amy.mat')
            targ_list = file_key.obstrialList
            targ_series = targ_list[targIx, :] - 63
            TC0 = x['targObjects'][:, :, int(targ_series[0])-1]
            TC2 = x['targObjects'][:, :, int(targ_series[2])-1]
            centerpos = np.array([ 0.0377292,  0.1383867])

            #### Get cneterpos; 
            centerPos = np.mean(TC0, axis=1)/100. + centerpos
            targetPos = np.mean(TC2, axis=1)/100. + centerpos

            #### Assume binsize is 0.1 (TODO: add this as an kwarg)
            centerPos = centerPos*20
            targetPos = targetPos*20
            
            ### Will return 0.0 or 0.1 depending
            cw_ccw, s = CW_CCW_obs(centerPos, targetPos, cursor_pos.T)

            if day_ix == 3 and targIx == 2:
                if s < 0.1: 
                    cw_ccw = 0.1
                    print('Correcting JEEV target 2 %.1f' %(s))
                else:
                    print('Keeping JEEV target 2 %.1f' %(s))

        return targIx + cw_ccw

    elif animal == 'grom':
        dats = sio.loadmat(analysis_config.config['grom_pref'] + 'unique_targ.mat')
        targetPos = np.squeeze(dats['unique_targ'][int(targIx), :])
        centerPos = np.array([0., 0.])
        cw_ccw, s = CW_CCW_obs(centerPos, targetPos, cursor_pos.T)
        return targIx + cw_ccw

def CW_CCW_obs(centerPos, targPos, trialPos):
    """
    Return 0: if CW to get to target, return 1 if CCW to get to target; 
    Steps: 
        1. Center by the centerPos
        2. Rotate by targPos angle such that aligned with (0, 1) axis; 
        3. Compute integral unneath curve; 

    Args:
        centerPos (np.array): (x, y) of centerPos (or start target for Jeev)
        targPos (np.array): (x, y) of targetPos (or end target for Jeev)
        trialPos (np.array): (2 x T) of trial trajectory (position)
    
    Returns:
        integer: 0 for CW, 1 for CCW
    """

    targCentered = targPos - centerPos
    targCentered_norm = targCentered / np.linalg.norm(targCentered)

    assert(trialPos.shape[0] == len(centerPos) == 2)
    trialCentered = trialPos - centerPos[:, np.newaxis]

    ### Rotate Target / trial; 
    angle = np.arctan2(targCentered_norm[1], targCentered_norm[0])

    ### Rotate by negative of angle: 
    Rot = np.array([[np.cos(-1*angle), -np.sin(-1*angle)], [np.sin(-1*angle), np.cos(-1*angle)]])

    ### Apply this rotation to the trial: 
    trialCentRot = np.dot(Rot, trialCentered)

    ### Get step size; 
    dx = np.hstack(([0], np.diff(trialCentRot[0, :])))

    ### Get the height: 
    y = trialCentRot[1, :]

    assert(len(dx) == len(y))

    if np.dot(dx, y) > 0:
        rot = 'CW'
        return 0.0, np.dot(dx, y)

    elif np.dot(dx, y) < 0:
        rot = 'CCW'
        return 0.1, np.dot(dx, y)

#### Extract Data ###
def get_grom_decoder(day_ix):
    co_obs_dict = pickle.load(open(analysis_config.config['grom_pref']+'co_obs_file_dict.pkl'))
    input_type = analysis_config.data_params['grom_input_type']

    ### First CO task for that day: 
    te_num = input_type[day_ix][0][0]
    dec = co_obs_dict[te_num, 'dec']
    decix = dec.rfind('/')
    decoder = pickle.load(open(analysis_config.config['grom_pref']+dec[decix:]))
    
    F, KG = decoder.filt.get_sskf()
    return F, KG

def get_jeev_decoder(day_ix):
    kgs = pickle.load(open(analysis_config.config['jeev_pref']+'jeev_KG_approx_fit.pkl', 'rb'))
    KG = kgs[day_ix]
    KG_potent = KG.copy(); #$[[3, 5], :]; # 2 x N
    return np.squeeze(np.array(KG_potent))

def get_jeev_F(day_ix):
    kgs = pickle.load(open(analysis_config.config['jeev_pref']+'jeev_KG_approx_fit.pkl', 'rb'))
    F = kgs[day_ix, 'F'] ## 7 x N 
    return np.squeeze(np.array(F))

def get_decoder(animal, day_ix):
    ''' 
    returns 2 x N decoder 
    '''
    if animal == 'grom':
        _, KG = get_grom_decoder(day_ix)
        return KG[[3, 5], :]
    elif animal == 'jeev':
        return get_jeev_decoder(day_ix)

def get_data_from_shuff(animal, day_ix, w_intc = True):
    dat = pickle.load(open(analysis_config.config['shuff_fig_dir']+'%s_%d_shuff_ix.pkl'%(animal, day_ix), 'rb'))
    
    #### Extract the data ####
    spks = dat['Data']['spks']
    push = dat['Data']['push']
    tsk = dat['Data']['task']
    trg = dat['Data']['targ']
    move = tsk*10 + trg
    bin_num = dat['Data']['bin_num']
    
    ### Get reversed bin number ####
    rev_bin_num = bin_num_relative_to_end(bin_num)
    bin_num = np.hstack((bin_num))
    
    ### Make sure all the samse length 
    assert(spks.shape[0] == push.shape[0] == len(tsk) == len(trg) == len(bin_num) == len(rev_bin_num))
    return spks, push, tsk, trg, bin_num, rev_bin_num, move, dat


#### Linear mixed effect modeling: 
def run_LME(Days, Grp, Metric, bar_plot = False, xlabels = None, title = ''):
    Days = np.hstack((Days))
    Grp = np.hstack((Grp))
    Metric = np.hstack((Metric))

    data = pd.DataFrame(dict(Days=Days, Grp=Grp, Metric=Metric))
    md = smf.mixedlm("Metric ~ Grp", data, groups=data["Days"])
    mdf = md.fit()
    pv = mdf.pvalues['Grp']
    slp = mdf.params['Grp']
    print('PV = {:.3e}'.format(pv)) 
    print('SLP %.5f, N = %d, t(%d) = %.3f' %(slp, len(Days), mdf.df_resid, mdf.tvalues['Grp']))

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
            ax.plot(ix, tmp, '-', color='gray', linewidth=.5)

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
        ax.set_xlim([-.5, 3.5])
        f.tight_layout()

    return pv, slp

def get_R2(y_true, y_pred, pop = True, ignore_nans = False):
    assert(y_true.shape == y_pred.shape)

    if np.logical_and(len(y_true.shape) == 1, len(y_pred.shape) == 1):
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]
    
    if ignore_nans:
        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.nansum(np.square(y_true - y_pred), axis=0)
        SST_i = np.nansum(np.square(y_true - np.nanmean(y_true, axis=0)[np.newaxis, :]), axis=0)

        if pop:
            return 1 - np.nansum(SSR_i)/np.nansum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)
    else: 

        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.sum(np.square(y_true - y_pred), axis=0)
        SST_i = np.sum(np.square(y_true - np.mean(y_true, axis=0)[np.newaxis, :]), axis=0)

        if pop:
            return 1 - np.sum(SSR_i)/np.sum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)

def get_VAF_no_mean(y_true, y_pred, pop = True, ignore_nans = False):
    assert(y_true.shape == y_pred.shape)

    if np.logical_and(len(y_true.shape) == 1, len(y_pred.shape) == 1):
        y_true = y_true[:, np.newaxis]
        y_pred = y_pred[:, np.newaxis]
    
    if ignore_nans:
        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.nansum(np.square(y_true - y_pred), axis=0)
        SST_i = np.nansum(np.square(y_true), axis=0)

        if pop:
            return 1 - np.nansum(SSR_i)/np.nansum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)
    else: 

        ### Assume Y-true, y_pred are T x N matrices: 
        SSR_i = np.sum(np.square(y_true - y_pred), axis=0)
        SST_i = np.sum(np.square(y_true), axis=0)

        if pop:
            return 1 - np.sum(SSR_i)/np.sum(SST_i)
        else:
            return 1 - (SSR_i / SST_i)

def savefig(f, name, png=False):
    '''
    save figure 
    '''
    if png:
        f.savefig(analysis_config.config['fig_dir'] + name + '.png')
    else:
        f.savefig(analysis_config.config['fig_dir'] + name + '.svg')

def rgba2rgb(rgba, bg_rgba=np.array([1., 1., 1.])):
    assert(np.all(rgba>=0.))
    assert(np.all(rgba<=1.))

    alpha = rgba[3]
    
    rnew = (1-alpha)*bg_rgba[0] + alpha*rgba[0]
    gnew = (1-alpha)*bg_rgba[1] + alpha*rgba[1]
    bnew = (1-alpha)*bg_rgba[2] + alpha*rgba[2]
    return np.array([rnew, gnew, bnew])

def get_color(mov, alpha=None):
    '''
    Put in the movement identifier to get color 
    '''
    colnm = analysis_config.pref_colors[int(mov)%10]
    colrgba = np.array(mpl_colors.to_rgba(colnm))

    ### Set alpha according to task (tasks 0-7 are CO, tasks 10.0 -- 19.1 are OBS) 
    if alpha is None:
        if mov >= 10:
            colrgba[-1] = 0.5
        else:
            colrgba[-1] = 1.0
    else:
        colrgba[-1] = alpha
    
    #### col rgb ######
    colrgb = rgba2rgb(colrgba)
    return colrgb


### Plotting ###
def draw_plot(xax, data, edge_color, fill_color, ax, width = .5, skip_median=False):
    bp = ax.boxplot(data, patch_artist=True, positions = [xax], widths=[width],
        whis = [5, 95])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    if skip_median:
        plt.setp(bp['medians'], color=np.array([1., 1., 1., 0.]))

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

def get_pv_str(pv):
    if pv < 0.001:
        pv_str = '***'
    elif pv < 0.01:
        pv_str = '**'
    elif pv < 0.05:
        pv_str = '*'
    else:
        pv_str = 'n.s.\npv=%.2f'%(pv)
    return pv_str

### Run PCA ###
def PCA(X, nPCs, mean_subtract = True, skip_dim_assertion = False):
    '''
    X is a T x N data matrix; 
    '''
    if skip_dim_assertion:
        print('Number of dim %d, number of obs %d' %(X.shape[1], X.shape[0]))
    else:
        assert(X.shape[0] > X.shape[1])
    
    ### assumes X is time x dimensions

    if mean_subtract:
        x_mn = np.mean(X, axis=0)
        assert(len(x_mn) == X.shape[1])
        X = X - x_mn[np.newaxis, :]
    else:
        x_mn = np.zeros((X.shape[1], ))

    covX = np.cov(X.T)
    assert(covX.shape[0] == covX.shape[1] == X.shape[1])

    ### Now do eig value decomp; 
    ev, vect = np.linalg.eig(covX)

    ### Chekc that acutally eigenvalues / vectors; 
    chk_ev_vect(ev, vect, covX)

    ### Ok now order them; 
    ix_sort = np.argsort(ev)
    evsort = ev[ix_sort[::-1]]
    assert(np.max(evsort) == evsort[0])
    vectsort = vect[:, ix_sort[::-1]]

    ### Check again; 
    chk_ev_vect(evsort, vectsort, covX)

    ### now find num PCs needed: 
    cumsum = np.cumsum(evsort)/np.sum(evsort)

    ### Transform data; 
    ### n x k PCs
    proj_mat = vectsort[:, :nPCs]

    ### k PCs x n  DOT n x T --> k x T (trans) --> T x k; 
    transf_data = np.dot(proj_mat.T, X.T).T
    assert(transf_data.shape[0] == X.shape[0])
    assert(transf_data.shape[1] == nPCs)

    pc_model = dict(proj_mat = proj_mat, x_mn = x_mn)

    return transf_data, pc_model, evsort

def dat2PC(X, pc_model):
    assert(X.shape[1] == len(pc_model['x_mn']))
    X_dmn = X - pc_model['x_mn'][np.newaxis, :]
    proj_mat= pc_model['proj_mat']
    return np.dot(proj_mat.T, X.T).T

### Double check that each eigenvlaue / vecotr is correct: 
def chk_ev_vect(ev2, vect2, covX2):
    for i in range(len(ev2)):
        evi = ev2[i]
        vct = vect2[:, i]
        ## Do out the multiplication
        assert(np.allclose(np.dot(covX2, vct[:, np.newaxis]), evi*vct[:, np.newaxis]))
