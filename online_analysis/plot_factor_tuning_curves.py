import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime; 
import pickle, os
from db import dbfunctions as dbfn
from statsmodels.formula.api import ols
import pandas
import co_obs_tuning_matrices
import analysis_config
import fit_LDS
from resim_ppf import file_key 
import prelim_analysis as pa
from resim_ppf import ppf_pa
try:
    from scipy.stats import nanmean
except:
    from numpy import nanmean
import tables 
import pandas
import seaborn
seaborn.set(font='Arial',context='talk',font_scale=1.5, style='white')

import scipy.io as sio
import scipy.stats
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import gc
from sklearn.linear_model import Ridge

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black']
cmap = cmap_list
blue_cmap = ['black', 'royalblue', 'lightsteelblue']
grey_cmap = np.flipud(np.array([[217,217,217], [189,189,189], [150,150,150], [99,99,99], [37,37,37]]))

co_obs_cmap = [np.array([0, 103, 56])/255., np.array([46, 48, 146])/255., ]
co_obs_cmap_cm = []; 

model_cols = [[255, 0, 0], [101, 44, 144], [39, 169, 225],]
model_cols = [np.array(m)/255. for m in model_cols]

for _, (c, cnm) in enumerate(zip(co_obs_cmap, ['co', 'obs'])):
    colors = [[1, 1, 1], c]  # white --> color
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(
        cnm, colors, N=1000)
    co_obs_cmap_cm.append(cm)

import co_obs_tuning_matrices, subspace_overlap
from resim_ppf import file_key 
grom_input_type = analysis_config.data_params['grom_input_type']
jeev_input_type = file_key.task_filelist

pref = analysis_config.config['grom_pref']
n_entries_hdf = 800;

fig_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

class Model_Table(tables.IsDescription):
    param = tables.Float64Col(shape=(n_entries_hdf, ))
    pvalue = tables.Float64Col(shape=(n_entries_hdf, ))
    r2 = tables.Float64Col(shape=(1,))
    aic = tables.Float64Col(shape=(1,))
    bic = tables.Float64Col(shape=(1,))
    day_ix = tables.Float64Col(shape=(1,))

def bin_spks(spks, update_bmi_ix):
    #Need to use 'update_bmi_ix' from ReDecoder to get bin edges correctly:
    nbins = len(update_bmi_ix)
    bin_spk_i = np.zeros((nbins, spks.shape[1]))

    for ib, i_ix in enumerate(update_bmi_ix):
        #Inclusive of EndBin
        bin_spk_i[ib,:] = np.sum(spks[i_ix-5:i_ix+1,:], axis=0)

    return bin_spk_i

def add_params_to_mult(model, test_data=None, predict_key='spks',only_potent_predictor=False, KG_pot = None):
    if only_potent_predictor:
        raise Exception('Not yet implemented')

    if test_data is not None: 
        model_params = model.params.index.values; 
        X = []; N = len(test_data['tsk'])
        
        for m, mod in enumerate(model_params):
            if mod == 'Intercept':
                X.append(np.ones((N, 1)))
            else:
                X.append(test_data[mod][:, np.newaxis])
        X = np.hstack((X))
        Y = test_data[predict_key]
    else: 
        print 'USING TRAINING DATA FOR R2'
        # Use training data: 
        X = model.model.exog #Remove intercept
        Y = np.mat(model.model.endog)

    X2 = np.linalg.pinv(np.dot(X.T, X))
    pred = (np.mat(model.params).T*np.mat(X).T).T
    
    SSR = np.sum((np.array(pred - Y))**2, axis=0) 
    sse = SSR / float(X.shape[0] - X.shape[1]) ## Getting dofs: 
    SST = np.sum(np.array( Y - np.mean(Y, axis=0))**2, axis=0 )
    se = np.array([ np.sqrt(np.diagonal(sse[i] * X2)) for i in range(sse.shape[0]) ])

    model.t_ = model.params / se.T
    model.pvalues = 2 * (1 - scipy.stats.t.cdf(np.abs(model.t_), Y.shape[0] - X.shape[1]))

    ### Some SST maybe zero. How to deal with this? 
    if len(np.nonzero(SST == 0)[0]) > 0:
        SST[np.nonzero(SST==0)[0]] = SSR[np.nonzero(SST==0)[0]]
    model.rsquared = 1 - (SSR/SST)
    
    nobs2=model.nobs/2.0 # decimal point is critical here!
    llf = -np.log(SSR) * nobs2
    llf -= (1+np.log(np.pi/nobs2))*nobs2
    model.aic = -2 *llf + 2 * (model.df_model + model.k_constant)
    model.bic = -2 *llf + np.log(model.nobs) * (model.df_model + model.k_constant)
    model.coef_names = model.params[0].keys()

    return model, pred

def sklearn_mod_to_ols(model, test_data=None, x_var_names=None, predict_key='spks', only_potent_predictor=False, 
    KG_pot = None, fit_task_specific_model_test_task_spec = False):

    x_test = [];
    for vr in x_var_names:
        x_test.append(test_data[vr][: , np.newaxis])
    X = np.mat(np.hstack((x_test)))
    Y = np.mat(test_data[predict_key])

    assert(X.shape[0] == Y.shape[0])

    if fit_task_specific_model_test_task_spec:
        ix0 = np.nonzero(test_data['tsk'] == 0)[0]
        ix1 = np.nonzero(test_data['tsk'] == 1)[0]

        X0 = X[ix0, :]; Y0 = Y[ix0, :];
        X1 = X[ix1, :]; Y1 = Y[ix1, :]; 

        pred0 = np.mat(X0)*np.mat(model[0].coef_).T + model[0].intercept_[np.newaxis, :]
        pred1 = np.mat(X1)*np.mat(model[1].coef_).T + model[1].intercept_[np.newaxis, :]

        pred = pred0; 
        model1 = model[1]; 
        model =  model[0]; 
        model.X = X0; 
        model.y = Y0; 

    model.nneurons = model.y.shape[1]
    model.nobs = model.X.shape[0]

    if only_potent_predictor:
        X = np.dot(KG_pot, X.T).T
        print 'only potent predictor'

    pred = np.mat(X)*np.mat(model.coef_).T + model.intercept_[np.newaxis, :]
    
    SSR = np.sum((np.array(pred - Y))**2, axis=0) 
    sse = SSR / float(X.shape[0] - X.shape[1])
    SST = np.sum(np.array( Y - np.mean(Y, axis=0))**2, axis=0 )
    if len(np.nonzero(SST == 0)[0]) > 0:
        SST[np.nonzero(SST==0)[0]] = SSR[np.nonzero(SST==0)[0]]
    try:
        X2 = np.linalg.pinv(np.dot(X.T, X))
    except:
        X2 = np.zeros_like(np.dot(X.T, X))
        print 'wouldnt trust aic / bic'
    se = np.array([ np.sqrt(np.diagonal(sse[i] * X2)) for i in range(sse.shape[0]) ])

    model.t_ = np.mat(model.coef_) / se
    model.pvalues = 2 * (1 - scipy.stats.t.cdf(np.abs(model.t_), Y.shape[0] - X.shape[1]))
    model.rsquared = 1 - (SSR/SST)
    
    nobs2=model.nobs/2.0 # decimal point is critical here!
    llf = -np.log(SSR) * nobs2
    llf -= (1+np.log(np.pi/nobs2))*nobs2
    model.aic = -2 *llf + 2 * (X.shape[1] + 1)
    model.bic = -2 *llf + np.log(model.nobs) * (X.shape[1] + 1)
    
    if fit_task_specific_model_test_task_spec:
        return [model, model1], [pred0, pred1]
    else:
        return model, pred

def h5_add_model(h5file, model_v, day_ix, first=False, model_nm=None, test_data=None, 
    fold = 0., xvars = None, predict_key='spks', only_potent_predictor = False, 
    KG_pot = None, fit_task_specific_model_test_task_spec = False):
    
    try:
        # OLS models: 
        nneurons = model_v.predict().shape[1]
        model_v, predictions = add_params_to_mult(model_v, test_data, predict_key, only_potent_predictor, KG_pot)
    except:
        # CLF/RIDGE models: 
        model_v, predictions = sklearn_mod_to_ols(model_v, test_data, xvars, predict_key, only_potent_predictor, KG_pot,
            fit_task_specific_model_test_task_spec)
        try:
            nneurons = model_v.nneurons
        except:
            nneurons = model_v[0].nneurons
    if first:
        tab = h5file.createTable("/", model_nm+'_fold_'+str(int(fold)), Model_Table)
        col = h5file.createGroup(h5file.root, model_nm+'_fold_'+str(int(fold))+'_nms')
        try:
            vrs = np.array(model_v.coef_names, dtype=np.str)
            h5file.createArray(col, 'vars', vrs)
        except:
            print 'skippign adding varaible names'
    else:
        tab = getattr(h5file.root, model_nm+'_fold_'+str(int(fold)))
    
    print nneurons, model_nm, 'day: ', day_ix

    for n in range(nneurons):
        row = tab.row
    
        #Add params: 
        #vrs = getattr(getattr(h5file.root, model_nm+'_fold_'+str(int(fold))+'_nms'), 'vars')[:]
        param = np.zeros((n_entries_hdf, ))
        pv = np.zeros((n_entries_hdf, ))
        for iv, v in enumerate(xvars):
            try:
                # OLS: 
                param[iv] = model_v.params[n][v]
                pv[iv] = model_v.pvalues[iv, n]
            except:
                # RIDGE: 
                if fit_task_specific_model_test_task_spec:
                    param[iv] = model_v[0].coef_[n, iv]
                    pv[iv] = model_v[0].pvalues[n, iv]
                else:
                    param[iv] = model_v.coef_[n, iv]
                    pv[iv] = model_v.pvalues[n, iv]

        row['param'] = param
        row['pvalue'] = pv
        row['day_ix'] = day_ix

        if fit_task_specific_model_test_task_spec:
            row['r2'] = model_v[0].rsquared[n]
            row['aic'] = model_v[0].aic[n]
            row['bic'] = model_v[0].bic[n]
        else:
            row['r2'] = model_v.rsquared[n]
            row['aic'] = model_v.aic[n]
            row['bic'] = model_v.bic[n]
        
        row.append()

    return h5file, model_v, predictions

def task_dict_add_model(task_dict_file, data_temp_dict, model_nms, pos_model_nms, name, i_d):
    tmp_x = []
    for x in model_nms:
        tmp_x.append(data_temp_dict[x])
    for x in pos_model_nms:
        tmp_x.append(data_temp_dict[x])

    X = np.vstack((tmp_x)).T
    Y = data_temp_dict['tsk']

    # Equalize input numbers:
    ix0 = np.nonzero(Y==0)[0]
    ix1 = np.nonzero(Y==1)[0]
    L = np.min([len(ix0), len(ix1)])

    ix0_ = np.random.permutation(len(ix0))[:L]
    ix1_ =  np.random.permutation(len(ix1))[:L]
    ix_new = np.hstack((ix0[ix0_], ix1[ix1_]))

    X_trim = X[ix_new, :]
    Y_trim = Y[ix_new]

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_trim, Y_trim)

    task_dict_file[name, i_d] = clf.predict_proba(X)
    task_dict_file[name, i_d, 'X'] = X
    task_dict_file[name, i_d, 'Y'] = Y
    task_dict_file[name, i_d, 'scale'] = clf.scalings_
    task_dict_file[name, i_d, 'int'] = clf.intercept_
    task_dict_file[name, i_d, 'coef'] = clf.coef_
    task_dict_file[name, i_d, 'score'] = clf.score(X, Y)
    return task_dict_file



### test data_temp ###
def plot_data_temp(data_temp, use_bg = False):

    ### for each target new plto: 
    fco, axco = plt.subplots()
    fob, axob = plt.subplots()
    for ax in [axco, axob]:
        ax.axis('square')
    tgs = [4, 5]

    ### open target: 
    for i in range(2): 
        tsk_ix = np.nonzero(data_temp['tsk'] == i)[0]
        targs = np.unique(data_temp['trg'][tsk_ix])

        for itr, tr in enumerate(targs):
            ## Get the trial numbers: 
            targ_ix = np.nonzero(data_temp['trg'][tsk_ix] == tr)[0]
            trls = np.unique(data_temp['trl'][tsk_ix[targ_ix]])

            if tgs[i] == itr:
                alpha = 1.0; LW = 2.0
            else:
                alpha = 0.4; LW = 1.0

            if i == 0:
                axi = axco; 
            else:
                axi = axob#[itr/3, itr%3]

            for trl in trls:
                ix = np.nonzero(data_temp['trl'][tsk_ix] == trl)[0]
                if use_bg:
                    axi.plot(data_temp['posx_tm0'][tsk_ix[ix]], data_temp['posy_tm0'][tsk_ix[ix]], '-', color=co_obs_cmap[i],
                        linewidth = LW, alpha=alpha)
                else:
                    axi.plot(data_temp['posx_tm0'][tsk_ix[ix]], data_temp['posy_tm0'][tsk_ix[ix]], '-', color=cmap_list[itr])

    ### Add target info for Grom: 
    for i, a in enumerate(np.linspace(0., 2*np.pi, 9)):
        if i < 8:
            tg = [10*np.cos(a), 10*np.sin(a)]
            circle = plt.Circle(tg, radius=1.7, color = cmap_list[i], alpha=.2)
            axob.add_artist(circle)
            circle = plt.Circle(tg, radius=1.7, color = cmap_list[i], alpha=.2)
            axco.add_artist(circle)

        for ax in [axco, axob]:
           ax.set_xlim([-12, 12])
           ax.set_ylim([-12, 12])
           ax.set_xticks([])
           ax.set_yticks([])

    # fco.savefig('co_eg.svg')
    # fob.savefig('ob_eg.svg')

def panda_to_dict(D):
    d = dict()
    for k in D.keys():
        d[k] = np.array(D[k][:])
    return d

def get_KG_decoder_grom(day_ix):
    co_obs_dict = pickle.load(open(co_obs_tuning_matrices.pref+'co_obs_file_dict.pkl'))
    input_type = analysis_config.data_params['grom_input_type']

    ### First CO task for that day: 
    te_num = input_type[day_ix][0][0]
    dec = co_obs_dict[te_num, 'dec']
    decix = dec.rfind('/')
    decoder = pickle.load(open(co_obs_tuning_matrices.pref+dec[decix:]))
    F, KG = decoder.filt.get_sskf()
    KG_potent = KG[[3, 5], :]; # 2 x N
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)

    ## Get KG potent too; 
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh)
    Va[:2, :] = Vh[:2, :]
    KG_potent_orth = np.dot(Va.T, Va)

    return KG_potent, KG_null_proj, KG_potent_orth

def generate_KG_decoder_jeev():
    KG_approx = dict() 

    ### Task filelist ###
    filelist = file_key.task_filelist
    days = len(filelist)
    binsize_ms = 5.
    important_neuron_ix = dict()

    for day in range(days):

        # open the thing: 
        Ps = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_feb_2019_day'+str(day)+'.pkl'))
        
        # Get bin spikes for this task entry:
        bin_spk_all = []
        cursor_KG_all = []
        P_mats = []

        for task in range(2):
            te_nums = filelist[day][task]
            
            for te_num in te_nums:
                bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all, unbinned, exclude = ppf_pa.get_jeev_trials_from_task_data(te_num, 
                    binsize=binsize_ms/1000.)

                indices = []
                for j, (i0, i1) in enumerate(unbinned['ixs']):
                    indices.append(np.arange(i0, i1) - unbinned['start_index_overall'])
                
                for j in np.hstack((indices)):
                    P_mats.append(np.array(Ps[te_num][j]))

                print unbinned['start_index_overall'], i1, i1 -  unbinned['start_index_overall']

                bin_spk_all.extend([bs for ib, bs in enumerate(bin_spk) if ib not in exclude])
                cursor_KG_all.extend([kg for ik, kg in enumerate(decoder_all) if ik not in exclude])

        bin_spk_all = np.vstack((bin_spk_all))
        P_mats = np.dstack((P_mats))

        cursor_KG_all = np.vstack((cursor_KG_all))[:, [3, 5]]
        
        # De-meaned cursor: 
        cursor_KG_all_demean = (cursor_KG_all - np.mean(cursor_KG_all, axis=0)[np.newaxis, :]).T
        
        # Fit a matrix that predicts cursor KG all from binned spike counts
        KG = np.mat(np.linalg.lstsq(bin_spk_all, cursor_KG_all)[0]).T
        
        cursor_KG_all_reconst = KG*bin_spk_all.T
        vx = np.sum(np.array(cursor_KG_all[:, 0] - cursor_KG_all_reconst[0, :])**2)
        vy = np.sum(np.array(cursor_KG_all[:, 1] - cursor_KG_all_reconst[1, :])**2)
        R2_best = 1 - ((vx + vy)/np.sum(cursor_KG_all_demean**2))
        print 'KG est: shape: ', KG.shape, ', R2: ', R2_best

        KG_approx[day] = KG.copy();
        KG_approx[day, 'R2'] = R2_best;

    ### Save this: 
    pickle.dump(KG_approx, open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_fit.pkl', 'wb'))

def get_KG_decoder_jeev(day_ix):
    kgs = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_KG_approx_fit.pkl', 'rb'))
    KG = kgs[day_ix]
    KG_potent = KG.copy(); #$[[3, 5], :]; # 2 x N
    KG_null = scipy.linalg.null_space(KG_potent) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)
    
    U, S, Vh = scipy.linalg.svd(KG_potent); #[2x2, 2, 44x44]
    Va = np.zeros_like(Vh)
    Va[:2, :] = Vh[:2, :]
    KG_potent_orth = np.dot(Va.T, Va)

    return KG_potent, KG_null_proj, KG_potent_orth

def get_decomp_y(KG_null, KG_pot, y_true, only_null = True, only_potent=False):
    if only_null:
        y_proj = np.dot(KG_null, y_true.T).T 

    elif only_potent:
        y_proj = np.dot(KG_pot, y_true.T).T

    ## Make sure kg*y_null is zero
    #np_null = np.dot(KG, y_null.T).T
    #f,ax = plt.subplots()
    #plt.plot(np_null[:, 0])
    #plt.plot(np_null[:, 1])
    #plt.title('Null Proj throuhg KG')

    # Do SVD on KG potent and ensure match to KG*y_true? 
    #N = y_true.shape[1]
    #U, S, Vh = scipy.linalg.svd(KG.T); #[2x2, 2, 44x44]

    # Take rows to Vh to make a projection basis: 
    #y_test = np.dot(KG, np.dot(U, np.dot(U.T, y_true.T)))
    #y_test2 = np.dot(KG, y_true.T)

    # f, ax = plt.subplots(ncols = 2)
    # ax[0].plot(y_test[0, :], y_test2[0,:], '.')
    # ax[1].plot(y_test[1, :], y_test2[1,:], '.')

    return y_proj; 

### Where to manipulate neural lags ###
def lag_ix_2_var_nm(lag_ixs, pos_or_vel='vel', nneur=0, neur_lag = 0, include_action_lags=False):
    nms = []
    if pos_or_vel == 'psh':
        if include_action_lags:
            for l in lag_ixs:
                if l<=0:
                    nms.append(pos_or_vel+'x_tm'+str(np.abs(l)))
                    nms.append(pos_or_vel+'y_tm'+str(np.abs(l)))
                else:
                    nms.append(pos_or_vel+'x_tp'+str(np.abs(l)))
                    nms.append(pos_or_vel+'y_tp'+str(np.abs(l)))                                  
        else:
            nms.append(pos_or_vel+'x_tm0')
            nms.append(pos_or_vel+'y_tm0')

    elif pos_or_vel == 'neur':
        for nl in neur_lag:
            if nl <= 0:
                t = 'm'
            elif nl > 0:
                t = 'p'
            for n in range(nneur):
                nms.append('spk_t'+t+str(int(np.abs(nl)))+'_n'+str(n))

    elif pos_or_vel == 'tg':
        nms.append('trg_posx')
        nms.append('trg_posy')
    else:
        for l in lag_ixs:
            if l <=0: 
                nms.append(pos_or_vel+'y_tm'+str(np.abs(l)))
                nms.append(pos_or_vel+'x_tm'+str(np.abs(l)))
            else:
                nms.append(pos_or_vel+'y_tp'+str(np.abs(l)))
                nms.append(pos_or_vel+'x_tp'+str(np.abs(l)))            
    
    nm_str = ''
    for s in nms:
        nm_str = nm_str + s + '+'
    return nms, nm_str[:-1]

def compute_residuals(model, ridge):
    if ridge:
        X = np.mat(model.X)
        Y = np.mat(model.y)
        pred = X*(model.coef_.T) + model.intercept_[np.newaxis, :]

    else:
        X = model.model.exog #Remove intercept
        Y = np.mat(model.model.endog)
        pred = (np.mat(model.params).T*np.mat(X).T).T
    
    residuals = Y - pred
    return residuals

### Main tuning function -- run this for diff animals; 
def model_individual_cell_tuning_curves(hdf_filename='_models_to_pred_mn_diffs', animal='grom', history_bins_max=4, 
    only_vx0_vy0_tsk_mod=True, task_prediction=False, n_folds = 5, compute_res = False, ridge=True,
    norm_neur = False, include_action_lags = True, 
    return_models = True, model_set_number = 8, ndays = None,
    include_null_pot = False,
    only_potent_predictor = False,
    fit_task_specific_model_test_task_spec = False):

    if model_set_number == 1:
        # models_to_include = ['prespos_0psh_1spksm_0_spksp_0', # only action, no state
        #                      'prespos_1psh_1spksm_0_spksp_0', # Current state, action, no spks
        #                      'hist_fut_4pos_1psh_1spksm_0_spksp_0', # State/action and lags, no spks
        #                      'hist_fut_4pos_1psh_1spksm_1_spksp_0', # State/action and lags, only spks_{t-1}
        #                      'hist_fut_4pos_1psh_1spksm_1_spksp_1'] # State/action and lags, spks_{t-1} AND spks_{t+1}
        # predict_key = 'spks'
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                            'hist_1pos_-1psh_1spksm_0_spksp_0', 
                            'hist_1pos_1psh_1spksm_0_spksp_0',
                            'hist_1pos_2psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_1_spksp_0']
        predict_key = 'spks'
        include_action_lags = False; ### Only action at t = 0; 
        history_bins_max = 1; 

    elif model_set_number == 2:
        models_to_include =['prespos_0psh_0spksm_1_spksp_0',
                            'hist_1pos_0psh_1spksm_0_spksp_0', 
                            'hist_1pos_0psh_0spksm_1_spksp_0']
        predict_key= 'psh'

    elif model_set_number == 3:
        models_to_include = [
                            # 'hist_1pos_0psh_1spksm_0_spksp_0', 
                            # 'hist_1pos_0psh_1spksm_1_spksp_0',
                            # 'hist_4pos_0psh_1spksm_0_spksp_0', 
                            # 'hist_4pos_0psh_1spksm_1_spksp_0',
                            # 'hist_4pos_0psh_1spksm_4_spksp_0',
                            'prespos_0psh_0spksm_1_spksp_0',
                            'hist_1pos_0psh_0spksm_1_spksp_0',]
                            #'hist_1pos_0psh_1spksm_0_spksp_0'];
        predict_key= 'spks'
        history_bins_max = 1;

    elif model_set_number == 4:
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                       'prespos_0psh_1spksm_1_spksp_0',

                       'prespos_1psh_1spksm_0_spksp_0', 
                       'prespos_1psh_1spksm_1_spksp_0',
                       
                       'hist_1pos_1psh_1spksm_0_spksp_0', 
                       'hist_1pos_1psh_1spksm_1_spksp_0', 
                       
                       'hist_2pos_1psh_1spksm_0_spksp_0', 
                       'hist_2pos_1psh_1spksm_1_spksp_0']
        
        include_action_lags = False
        predict_key = 'spks'
        history_bins_max = 2; 

    elif model_set_number == 8:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_1psh_0spksm_0_spksp_0',
            'hist_1pos_3psh_0spksm_0_spksp_0']
        #models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']
        include_action_lags = False; 
        history_bins_max = 1; 
        predict_key = 'spks'


    '''
    Summary: 
    
    Modified 9/13/19 to add neural tuning on top of this; 
    Modified 9/16/19 to add model return on top of this; 
    Modiefied 9/17/19 to use specialized alphas for each model, and change notation to specify y_{t-1} and/or y_{t+1}

    Input param: input_type:same -- fk.task_filelist, input_type
    Input param: hdf_filename: name to save params to 
    Input param: animal: 'grom' or jeev'
    Input param: history_bins_max: number of bins to use into the past and 
        future to model spike counts (binsize = 100ms)
    Input param: only_vx0_vy0_tsk_mod: True if you only want to model res ~ velx_tm0:tsk + vely_tm0:tsk instead
        of all variables being task-modulated

    Input param: task_prediction: True if you want to include way to predict task (0: CO, 1: Obs) from kinematics
    Input param: **kwargs: 
        task_pred_prefix -- what is this? 

        use_lags: list of lags to actually use (e.g. np.arange(-4, 5, 2))
        normalize_neurons: whether to compute a within-day mFR and sdFR so that high FR neurons don't dominate R2 
            calculations
    Output param: 
    '''
    if animal == 'jeev':
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
    else:
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

    if hdf_filename is None: 
        x = datetime.datetime.now()
        hdf_filename = animal + '_' + x.strftime("%Y_%m_%d_%H_%M_%S") + '.h5'
        hdf_filename = pref + hdf_filename; 

    else:
        if fit_task_specific_model_test_task_spec:
            hdf_filename = pref + animal + hdf_filename + '_model_set%d_task_spec.h5' %model_set_number
        else:
            hdf_filename = pref + animal + hdf_filename + '_model_set%d.h5' %model_set_number

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    ridge_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    if animal == 'grom':
        order_dict = co_obs_tuning_matrices.ordered_input_type
        input_type = analysis_config.data_params['grom_input_type']

    elif animal == 'jeev':
        order_dict = file_key.ordered_task_filelist
        input_type = file_key.task_filelist

    if ndays is None:
        pass
    else:
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    if task_prediction:
        task_dict_file = {} 
        if 'task_pred_prefix' not in kwargs.keys():
            raise Exception

    mFR = {}
    sdFR = {}

    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, spikes, sub_spk_temp_all, sub_push_all = get_spike_kinematics(animal, day, 
            order_dict[i_d], history_bins_max)

        ### Get kalman gain etc. 
        if animal == 'grom':
            KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)

        elif animal == 'jeev':
            KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

        assert np.linalg.matrix_rank(KG_potent_orth) == 2

        ### Replicate datastructure for saving later: 
        if return_models:

            ### Want to save neural push, task, target 
            model_data[i_d, 'spks'] = spikes.copy();
            model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))
            model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
            model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))
            model_data[i_d, 'bin_num'] = np.squeeze(np.array(data_temp['bin_num']))
            model_data[i_d, 'pos'] = np.vstack((np.array(data_temp['posx_tm0']), np.array(data_temp['posy_tm0']))).T
            model_data[i_d, 'vel'] = np.vstack((np.array(data_temp['velx_tm0']), np.array(data_temp['vely_tm0']))).T
            model_data[i_d, 'vel_tm1'] = np.vstack((np.array(data_temp['velx_tm1']), np.array(data_temp['vely_tm1']))).T
            model_data[i_d, 'trl'] = np.squeeze(np.array(data_temp['trl']))
            
            ### Models -- save predicitons
            for mod in models_to_include:
                if model_set_number in [1, 3, 4, 8]:
                    model_data[i_d, mod] = np.zeros_like(spikes) 

                    if include_null_pot:
                        ### if just use null / potent parts of predictions and propogate those guys
                        model_data[i_d, mod, 'null'] = np.zeros_like(spikes)
                        model_data[i_d, mod, 'pot'] = np.zeros_like(spikes)


                elif model_set_number == 2:
                    model_data[i_d, mod] = np.zeros_like(np.squeeze(np.array(sub_push_all)))

        if norm_neur:
            print 'normalizing!'
            mFR[i_d] = np.mean(spikes, axis=0)
            sdFR[i_d] = np.std(spikes, axis=0)
            sdFR[i_d][sdFR[i_d]==0] = 1
            spikes = ( spikes - mFR[i_d][np.newaxis, :] ) / sdFR[i_d][np.newaxis, :]

        ########################################################################################################
        ### Ok, time to amp up model-wise : Fit all neurons simulataneously, and add many more model options ###
        ########################################################################################################
        ### Get training and testing datasets: 
        N_pts = [];
        for tsk in range(2):
            ix = np.nonzero(data_temp['tsk'] == tsk)[0]

            ### Shuffle the task indices
            N_pts.append(ix[np.random.permutation(len(ix))])
        
        train_ix = dict();
        test_ix = dict(); 

        for i_f, fold_perc in enumerate(np.arange(0, 1., 1./n_folds)):
            test_ix[i_f] = []
            train_ix[i_f] = []; 

            for tsk in range(2):
                ntmp = len(N_pts[tsk])
                tst = N_pts[tsk][int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
                trn = np.array([j for i, j in enumerate(N_pts[tsk]) if j not in tst])
                
                test_ix[i_f].append(tst)
                train_ix[i_f].append(trn)

            test_ix[i_f] = np.hstack((test_ix[i_f]))
            train_ix[i_f] = np.hstack((train_ix[i_f]))

            tmp = np.unique(np.hstack((test_ix[i_f], train_ix[i_f])))

            ### Make sure that unique point is same as all data
            assert len(tmp) == len(data_temp)

        for i_fold in range(n_folds):

            # Useless. 
            # data_dict = panda_to_dict(data)
            # data_dict['spks'] = spikes

            ### TEST DATA ####
            data_temp_dict_test = panda_to_dict(data_temp.iloc[test_ix[i_fold]])
            data_temp_dict_test['spks'] = spikes[test_ix[i_fold]]
            data_temp_dict_test['pshy'] = sub_push_all[test_ix[i_fold], 1]
            data_temp_dict_test['pshx'] = sub_push_all[test_ix[i_fold], 0]
            data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

            ### TRAIN DATA ####
            data_temp_dict = panda_to_dict(data_temp.iloc[train_ix[i_fold]])
            data_temp_dict['spks'] = spikes[train_ix[i_fold]]
            data_temp_dict['pshy'] = sub_push_all[train_ix[i_fold], 1]
            data_temp_dict['pshx'] = sub_push_all[train_ix[i_fold], 0]
            data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))

            nneur = sub_spk_temp_all.shape[2]

            ##############################################
            ################# RUNNING ALL THE MODELS ###
            ##############################################
            # Velocity models: number of lags:

            ### NEW  ###
            model_var_list = []
            for i in np.arange(-history_bins_max, history_bins_max+1, 1):
                if i < 0:
                    model_var_list.append([np.arange(i, 1), 'hist_'+str(abs(i))])
                elif i == 0:
                    model_var_list.append([np.array([0]), 'pres'])
                elif i > 0:
                    model_var_list.append([np.arange(0, i+1), 'fut_'+str(i)])
            
            ### Add combo history/future methods: 
            for i in np.arange(1, history_bins_max + 1):
                model_var_list.append([np.arange(-1*i, i+1), 'hist_fut_'+str(i)])

            if models_to_include is not None:
                ### Figure out a way to overlook the above models: 
                ### go through the above models and keep if the nm shows up in models-to-include: 
                model_var_list2 = []; already_added = []; 
                for _, (lags, nm) in enumerate(model_var_list):
                    for mod in models_to_include:
                        if nm in mod:
                            if nm not in already_added:
                                model_var_list2.append([lags, nm])
                                already_added.append(nm)

                ### Manually remove future only: 
                model_var_list = []; 
                for _, (lags, mod) in enumerate(model_var_list2):
                    if np.logical_and('fut' in mod, len(mod) < 7):
                        pass
                    else:
                        model_var_list.append([lags, mod])
                print('Manually removing future. Dont panic')

            if model_set_number in [2, 3, 4]:
                print('Manually removing current time point as predictor except if pres')
                model_var_list_ = []; 
                for _, (lags, mod) in enumerate(model_var_list):
                    if 'pres' in mod:
                        pass
                    else:
                        if len(np.nonzero(lags==0)[0]) > 0:
                            ix = np.nonzero(lags != 0)[0]
                            lags = lags[ix]; 
                    model_var_list_.append([lags, mod])
                model_var_list = model_var_list_
                
                add_state_range = [0]; ### No state
                add_push_range = [0, 1];
                add_neg_neur_range = range(2); 
                add_pos_neur_range = range(1); # no positive neural activity pls. 

            elif model_set_number == 1:
                add_state_range = range(-1, 4);  ### All sorts of states!
                add_push_range = [1]; ### Everything should have push! 
                add_neg_neur_range = range(2); 
                add_pos_neur_range = range(1); # no future neural activvty
                model_var_list = [[np.array([-1]), 'hist_1'], [np.array([0]), 'pres'],]

            if model_set_number == 4:
                add_state_range = range(2)
                add_push_range = [1]; 
                add_neg_neur_range = range(2);
                add_pos_neur_range = [0]; 
                model_var_list = [[np.array([-2]), 'hist_2'], [np.array([-1]), 'hist_1'], [np.array([0]), 'pres'], ]

            if model_set_number == 8:
                add_state_range = range(-1, 4);  ### All sorts of states!
                add_push_range = [0]
                add_neg_neur_range = range(2);
                add_pos_neur_range = [0]; 
                model_var_list = [[np.array([-1]), 'hist_1'], [np.array([0]), 'pres'], ]

            ## Go through the models
            models_added = []; 
            for im, (model_vars, model_nm) in enumerate(model_var_list):
                
                ### State tuning, velocity: 
                for add_state in add_state_range:
                    
                    if add_state == 1:
                        vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                        pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                        tg_model_nms = tsk_model_nms = []; 
                    
                    elif add_state == -1:
                        pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                        vel_model_nms = []; 
                        tg_model_nms = tsk_model_nms = []; 
                    
                    elif add_state == 0:
                        vel_model_nms = pos_model_nms = []; 
                        tg_model_nms = tsk_model_nms = []; 

                    elif add_state == 2:
                        pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                        vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                        tg_model_nms, _ = lag_ix_2_var_nm(model_vars, 'tg')
                        tsk_model_nms = []; 

                    elif add_state == 3:
                        pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                        vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                        tg_model_nms, _ = lag_ix_2_var_nm(model_vars, 'tg')
                        tsk_model_nms = 'tsk'                    

                    ### Use push: 
                    for add_push in add_push_range: 
                        if add_push == 0:
                            push_model_nms = []; 

                        elif add_push == 1:
                            ### This will use the lags to add push; great. 
                            push_model_nms, push_model_str = lag_ix_2_var_nm(model_vars, 'psh', 
                                include_action_lags=include_action_lags)  

                        ### Add positive neural activity: 
                        for add_neg_neural in add_neg_neur_range:

                            ### #NOTE THAT THIS WILL cycle through option of having many or only 1 neural lag ###
                            for n_negs in [model_vars]: #[[-1], model_vars]:

                                if add_neg_neural == 0:
                                    neur_nms0 = []; 
                                    n_negs_all = []; 
                                
                                elif add_neg_neural == 1:
                                    n_negs_all = n_negs; 
                                    neur_nms0, _ = lag_ix_2_var_nm(model_vars, 'neur', nneur, neur_lag=n_negs_all)
                                    
                                ### Add positive neural activity;  
                                for add_pos_neural in add_pos_neur_range:
                                    if add_pos_neural == 0:
                                        neur_nms1 = []; 
                                    else:
                                        neur_nms1, _ = lag_ix_2_var_nm(model_vars, 'neur', nneur, neur_lag=[1])

                                    #############################
                                    ### Model with parameters ###
                                    #############################
                                    # if len(pos_model_str) > 0:
                                    #     st = 'spks~'+model_str+'+'+pos_model_str
                                    # else:
                                    #     st = 'spks~'+model_str

                                    # ### Add push if needed: 
                                    # if len(push_model_str) > 0:
                                    #     st = st + '+' + push_model_str; 

                                    # if len(neur_model_str) > 0:
                                    #     st = st + '+' + neur_model_str; 

                                    variables = np.hstack((vel_model_nms, pos_model_nms, tg_model_nms, tsk_model_nms, push_model_nms,neur_nms0, neur_nms1))
                                
                                    if ridge:
                                        giant_model_name = model_nm + 'pos_'+str(add_state) + 'psh_'+str(add_push) +'spksm_'+str(len(n_negs_all))+'_spksp_'+str(add_pos_neural)
                                        save_model = False; 

                                        ### Dont add the prespos mdoel with tm1 variables; 
                                        skip = False;
                                        if model_set_number in [2, 3]:
                                            if np.logical_and(model_nm == 'pres', n_negs[0] == -1):
                                                skip = True; 
                                                #print 'Skipping %s, vars, '% giant_model_name
                                                #print variables
                                        if giant_model_name in models_to_include:
                                            ### Don't add doubel: 
                                            if np.logical_or(giant_model_name in models_added, skip):
                                                pass
                                            else:
                                                alpha_spec = ridge_dict[animal][0][i_d, giant_model_name]

                                                if giant_model_name == 'prespos_0psh_0spksm_1_spksp_0':
                                                    if predict_key == 'spks':
                                                        alpha_spec = 0; 
                                                print('Using alpha %f, adding model %s' %(alpha_spec, giant_model_name))
                                                model_ = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, alpha=alpha_spec, 
                                                    only_potent_predictor = only_potent_predictor, KG_pot = KG_potent_orth, 
                                                    fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec)

                                                models_added.append(giant_model_name)
                                                save_model = True; 
                                    
                                    else:
                                        raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                                        model_ = ols(st, data_temp_dict).fit()

                                    ### Important --- name the model: 
                                    #name = model_nm + 'pos_'+str(add_pos) + 'psh_'+str(add_push) +'spks_'+str(add_neural)
                                    
                                    if save_model:
                                        h5file, model_, pred_Y = h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=giant_model_name, 
                                            test_data = data_temp_dict_test, fold = i_fold, xvars = variables, predict_key=predict_key, 
                                            only_potent_predictor = only_potent_predictor, KG_pot = KG_potent_orth,
                                            fit_task_specific_model_test_task_spec = fit_task_specific_model_test_task_spec)

                                        ### Save models, make predictions ####
                                        ### Need to figure out which spikes are where:
                                        if fit_task_specific_model_test_task_spec:
                                            ix0 = np.nonzero(data_temp_dict_test['tsk'] == 0)[0]
                                            ix1 = np.nonzero(data_temp_dict_test['tsk'] == 1)[0]
                                            
                                            model_data[i_d, giant_model_name][test_ix[i_fold][ix0], :] = np.squeeze(np.array(pred_Y[0]))
                                            model_data[i_d, giant_model_name][test_ix[i_fold][ix1], :] = np.squeeze(np.array(pred_Y[1]))
    
                                        else:
                                            model_data[i_d, giant_model_name][test_ix[i_fold], :] = np.squeeze(np.array(pred_Y))
                                           
                                            ### Save model -- for use later. 
                                            if model_set_number == 8:
                                                model_data[i_d, giant_model_name, i_fold, 'model'] = model_; 
                                                model_data[i_d, giant_model_name, i_fold, 'model_testix'] = test_ix[i_fold]; 


                                        #### Add / null potent? 
                                        if include_null_pot:

                                            ### Get predictors together: 
                                            x_test = [];
                                            for vr in variables:
                                                x_test.append(data_temp_dict_test[vr][: , np.newaxis])
                                            X = np.mat(np.hstack((x_test)))

                                            ### Get null and potent -- KG_null_proj, KG_potent_orth
                                            X_null = np.dot(KG_null_proj, X.T).T
                                            X_pot =  np.dot(KG_potent_orth, X.T).T

                                            assert np.allclose(X, X_null + X_pot)

                                            ### Need to divide the intercept into null / potent: 
                                            intc = model_.intercept_
                                            intc_null = np.dot(KG_null_proj, intc)
                                            intc_pot = np.dot(KG_potent_orth, intc)

                                            assert np.allclose(np.sum(np.abs(np.dot(KG, intc_null))), 0)
                                            assert np.allclose(np.sum(np.abs(np.dot(KG, X_null.T))), 0)

                                            pred_null = np.mat(X_null)*np.mat(model_.coef_).T + intc_null[np.newaxis, :]
                                            pred_pot = np.mat(X_pot)*np.mat(model_.coef_).T + intc_pot[np.newaxis, :]

                                            assert np.allclose(pred_Y, pred_null + pred_pot)

                                            ### This just propogates the identity; 
                                            # model_data[i_d, giant_model_name][test_ix[i_fold], :] = X.copy()
                                            model_data[i_d, giant_model_name, 'null'][test_ix[i_fold], :] = pred_null.copy()
                                            model_data[i_d, giant_model_name, 'pot'][test_ix[i_fold], :] = pred_pot.copy()
                                            

                                    ### TASK PREDICTION -- ###
                                    if task_prediction:
                                        raise Exception('deprecated')
                                        task_dict_file = task_dict_add_model(task_dict_file, data_temp_dict, model_nms_pls_vxvy0, pos_model_nms, name, i_d)
                                        
                                    #########################
                                    ### Compute Residuals ###
                                    #########################
                                    if compute_res:
                                        raise Exception('deprecated')
                                        # Check task-related modulations: 
                                        residuals = compute_residuals(model_, ridge)
                                        res_key = 'res_'+name
                                        data_temp_dict[res_key] = residuals

                                        ##################################
                                        ### Model with Task parameters ###
                                        ##################################
                                        # Create string: 
                                        if only_vx0_vy0_tsk_mod:
                                            var = 'velx_tm0*tsk + vely_tm0*tsk'
                                        
                                        else:
                                            var = ''
                                            
                                            for vr in model_.params[0].keys():
                                                # Don't want intercept modulated by task, nor trial_ord modulated by task
                                                if np.logical_and(vr != 'Intercept', vr != 'trial_ord'):
                                                    var = var + vr + '*tsk +'
                                            
                                            var = var[:-1]
                                        
                                        if ridge and only_vx0_vy0_tsk_mod:
                                            data_temp_dict['tsk_vx'] = data_temp_dict['velx_tm0']*data_temp_dict['tsk']
                                            data_temp_dict['tsk_vy'] = data_temp_dict['vely_tm0']*data_temp_dict['tsk']
                                            model_2 = fit_ridge(residuals, data_temp_dict, ['tsk_vx', 'tsk_vy'])
                                        else:
                                            model_2 = ols(res_key + ' ~ '+var, data_temp_dict).fit()

                                        h5file, model_2 = h5_add_model(h5file, model_2, i_d, first=i_d==0, model_nm=res_key)
                            
                ################################
                ## Do model without position: ##
                ################################
                
                # st = 'spks~'+model_str
                # if ridge:
                #     model_ = fit_ridge(data_temp_dict['spks'], data_temp_dict, model_nms_pls_vxvy0)
                # else:
                #     model_ = ols(st, data_temp_dict).fit()

                # name = 'v'+str(lg)+'_'+str(st_ix)
                # h5file, model_ = h5_add_model(h5file, model_, i_d, first=i_d==0, model_nm=name)
                
                # residuals_2 = compute_residuals(model_, ridge)
                # res_key = 'res_'+name
                # data_temp_dict[res_key] = residuals_2

                # if only_vx0_vy0_tsk_mod:
                #     var = 'velx_tm0*tsk + vely_tm0*tsk'
                # else:
                #     var = ''
                #     for vr in model_.params[0].keys():
                #         # Don't want intercept modulated by task, nor trial_ord modulated by task
                #         if np.logical_and(vr != 'Intercept', vr != 'trial_ord'):
                #             var = var + vr + '*tsk +'
                #     var = var[:-1]
                
                # if ridge:
                #     if only_vx0_vy0_tsk_mod:
                #         data_temp_dict['tsk_vx'] = data_temp_dict['velx_tm0']*data_temp_dict['tsk']
                #         data_temp_dict['tsk_vy'] = data_temp_dict['vely_tm0']*data_temp_dict['tsk']        
                #         model_2 = fit_ridge(residuals_2, data_temp_dict, ['tsk_vx', 'tsk_vy'])
                # else:
                #     model_2 = ols(res_key + ' ~ '+var, data_temp_dict).fit()
                # h5file, model_2 = h5_add_model(h5file, model_2, i_d, first=i_d==0, model_nm=res_key)
                # if task_prediction:
                #     task_dict_file = task_dict_add_model(task_dict_file, data_temp_dict, model_nms, [], name, i_d)
        
        if task_prediction:
            nm = kwargs['task_pred_prefix']+'_day'+str(i_d)+'.mat'
            pickle.dump(task_dict_file, open(nm, 'wb'))
            print 'Task File Done: ', nm
            task_dict_file = {}

    h5file.close()
    print 'H5 File Done: ', hdf_filename

    ### ALSO SAVE MODEL_DATA: 
    if only_potent_predictor:
        pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_only_pot.pkl' %model_set_number, 'wb'))
    else:
        if fit_task_specific_model_test_task_spec:
            pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_task_spec.pkl' %model_set_number, 'wb'))
        else:
            pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'wb'))

def model_individual_cell_tuning_curves_task_spec_input(hdf_filename='_models_to_pred_mn_diffs_tsk_spec', 
    animal='grom', n_folds = 5, model_set_number = 6, ndays = 1, history_bins_max = 1,
    ridge = True, test_across_cond = False):

    ### Target info grom
    target_dats = sio.loadmat('/Users/preeyakhanna/fa_analysis/online_analysis/unique_targ.mat')
    target_dats = target_dats['unique_targ']

    if model_set_number == 5:
        models_to_include = ['hist_1pos_1psh_0spksm_1_spksp_0', # s_{t-1}, y_{t-1}
                             'hist_1pos_1psh_1spksm_1_spksp_0',]# s_{t-1}, y_{t-1}, a_t
        ### Fit data on all variables EXCEPT state 
        ### Then fit state for subset of task/target
        ### Held out data should be for that task/target percentage.                              
        predict_key= 'spks'

    elif model_set_number == 6:
        models_to_include = ['hist_1pos_1psh_0spksm_0_spksp_0'] ## only state. 
        predict_key = 'psh'

    if animal == 'jeev':
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
    else:
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

    if hdf_filename is None: 
        x = datetime.datetime.now()
        hdf_filename = animal + '_' + x.strftime("%Y_%m_%d_%H_%M_%S") + '.h5'
        hdf_filename = pref + hdf_filename; 
    else:
        hdf_filename = pref + animal + hdf_filename + '_model_set%d.h5' %model_set_number

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    ridge_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    if animal == 'grom':
        order_dict = co_obs_tuning_matrices.ordered_input_type
        input_type = analysis_config.data_params['grom_input_type']

    elif animal == 'jeev':
        order_dict = file_key.ordered_task_filelist
        input_type = file_key.task_filelist

    if ndays is None:
        pass
    else:
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, spikes, sub_spk_temp_all, sub_push_all = get_spike_kinematics(animal, day, 
            order_dict[i_d], history_bins_max)

        ### Replicate datastructure for saving later: 
        ### Want to save neural push, task, target 
        model_data[i_d, 'spikes'] = spikes.copy();
        model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))
        model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
        model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))

        ### Models -- save predicitons
        for mod in models_to_include:

            if predict_key == 'spks':
                model_data[i_d, mod] = np.zeros_like(spikes) 
                model_data[i_d, mod, 'common'] = np.zeros_like(spikes)

            elif predict_key == 'psh':
                model_data[i_d, mod] = np.zeros_like(model_data[i_d, 'np']) 
                model_data[i_d, mod, 'common'] = np.zeros_like(model_data[i_d, 'np'])

        ##########################################################################
        #### Fit all neurons simulataneously, and add many more model options ###
        #########################################################################
        ### Get training and testing datasets -- 
        train_ix = dict();
        test_ix = dict(); 
        train_ix_all = dict(); 

        nneur = sub_spk_temp_all.shape[2]
        T_all = len(model_data[i_d,'task'])

        for i_trg in range(8): 
            for i_t in range(2):
                print ("Targ %d, Task %d" %(i_trg, i_t))
                ### Get all indices that coudl be used to train task/target; 
                ix_all = (model_data[i_d, 'task'] == i_t) & (model_data[i_d, 'trg'] == i_trg); 
                ix_all = np.nonzero( ix_all == True )[0]

                ### Shuffle all this; 
                ix_all = ix_all[np.random.permutation(len(ix_all))]
                ntmp = float(len(ix_all))

                ### Split this up into train / test; 
                for i_f, fold_perc in enumerate(np.arange(0, 1., 1./n_folds)):
                    tst = ix_all[int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
                    trn = np.array([j for i, j in enumerate(ix_all) if j not in tst])
                    trn_all = np.array([j for j in range(T_all) if j not in tst])

                    test_ix[i_t, i_trg, i_f] = tst; 
                    train_ix[i_t, i_trg, i_f] = trn;
                    train_ix_all[i_t, i_trg, i_f] = trn_all; 

                    assert(len(tst) + len(trn) == ntmp)
                    assert(len(tst) + len(trn_all) == T_all)

                ### Clear how to split data: 
                for i_fold in range(n_folds):

                    ### TEST DATA ####
                    testing = test_ix[i_t, i_trg, i_fold]
                    data_temp_dict_test = panda_to_dict(data_temp.iloc[testing])
                    data_temp_dict_test['spks'] = spikes[testing]
                    data_temp_dict_test['pshx'] = sub_push_all[testing, 0]
                    data_temp_dict_test['pshy'] = sub_push_all[testing, 1]
                    data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))
                    
                    ### TRAIN DATA #### -- training for the task-specific. 
                    training = train_ix[i_t, i_trg, i_fold]
                    data_temp_dict = panda_to_dict(data_temp.iloc[training])
                    data_temp_dict['spks'] = spikes[training]
                    data_temp_dict['pshx'] = sub_push_all[training, 0]
                    data_temp_dict['pshy'] = sub_push_all[training, 1]
                    data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))

                    ### TRAIN DATA #### -- training for the GENERAL neural dynamics
                    training_all = train_ix_all[i_t, i_trg, i_fold]
                    data_temp_dict_all = panda_to_dict(data_temp.iloc[training_all])
                    data_temp_dict_all['spks'] = spikes[training_all]
                    data_temp_dict_all['pshx'] = sub_push_all[training_all, 0]
                    data_temp_dict_all['pshy'] = sub_push_all[training_all, 1]
                    data_temp_dict_all['psh'] = np.hstack(( data_temp_dict_all['pshx'], data_temp_dict_all['pshy']))

                    assert(len(testing) + len(training) == ntmp)
                    assert(len(testing) + len(training_all) == T_all)

                    ##############################################
                    ################# RUNNING ALL THE MODELS ###
                    ##############################################
                    # Velocity models: number of lags:
                    ### NEW  ###
                    model_var_list = [[np.array([-1]), 'hist_1']]

                    ## Go through the models
                    models_added = []; 
                    for im, (model_vars, model_nm) in enumerate(model_var_list):
                        
                        ### State tuning, velocity: 
                        for add_state in range(1, 2):
                            
                            if add_state == 1:
                                vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                                #print('Removing velocity')
                                vel_model_nms = []; 
                                pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                                
                                #print('adding target inof. ')
                                #pos_model_nms = np.hstack((pos_model_nms, 'trg'))

                            else:
                                vel_model_nms = pos_model_nms = []; 

                            ### Use push: 
                            for add_push in range(2): 
                                if add_push == 0:
                                    push_model_nms = []; 

                                elif add_push == 1:

                                    ### ONLY ADDING CURRENT PUSH
                                    push_model_nms, push_model_str = lag_ix_2_var_nm(model_vars, 'psh', 
                                        include_action_lags=False)  

                                ### Get neural lags
                                for add_neur in range(2):
                                    if add_neur == 1:
                                        neur_nms0, _ = lag_ix_2_var_nm(model_vars, 'neur', nneur, neur_lag=[-1])
                                    else:
                                        neur_nms0 = []; 

                                    variables = np.hstack((vel_model_nms, pos_model_nms, push_model_nms,
                                                neur_nms0))
                                    
                                    if ridge:
                                        giant_model_name = model_nm + 'pos_'+str(add_state) + 'psh_'+str(add_push) +'spksm_'+str(add_neur)+'_spksp_0'
                                        save_model = False


                                        if giant_model_name in models_to_include:

                                            ##### Get model alpha ######
                                            if giant_model_name == 'hist_1pos_1psh_0spksm_1_spksp_0':
                                                ### Has only spks.
                                                alpha_gen = ridge_dict[animal][0][i_d, 'hist_1pos_0psh_0spks_1_spksp_0']
                                                alpha_sta = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_0spks_0_spksp_0']
                                                alpha_com = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_0spks_1_spksp_0']
                                            #### This one has push: 
                                            elif giant_model_name == 'hist_1pos_1psh_1spksm_1_spksp_0':
                                                alpha_gen = ridge_dict[animal][0][i_d, 'hist_1pos_0psh_1spks_1_spksp_0']
                                                alpha_sta = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_0spks_0_spksp_0']
                                                alpha_com = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_1spks_1_spksp_0']

                                            elif giant_model_name == 'hist_1pos_1psh_0spksm_0_spksp_0':
                                                alpha_gen = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_0spks_0_spksp_0']
                                                alpha_sta = ridge_dict[animal][0][i_d, 'hist_1pos_1psh_0spks_0_spksp_0']
                                                alpha_com = None

                                            print('Using alphas %.2f, %.2f, adding model %s' %(alpha_gen, alpha_sta, giant_model_name))
                                            
                                            #### Get the model ###
                                            model1, model2, pred_Y, pred_Y_comm = fit_ridge_2step(data_temp_dict_all, data_temp_dict, data_temp_dict_test, 
                                                variables, alpha = alpha_gen, alpha_state = alpha_sta, alpha_common = alpha_com, 
                                                predict_key = predict_key)

                                            ### Add the giant mdoel name. 
                                            models_added.append(giant_model_name)
                                            save_model = True; 
                                    
                                    else:
                                        raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                                        model_ = ols(st, data_temp_dict).fit()

                                        ### Important --- name the model: 
                                        #name = model_nm + 'pos_'+str(add_pos) + 'psh_'+str(add_push) +'spks_'+str(add_neural)
                                        
                                    if save_model:
                                        ### Save models, make predictions ####
                                        ### Need to figure out which spikes are where:
                                        model_data[i_d, giant_model_name][testing, :] = np.squeeze(np.array(pred_Y))
                                        model_data[i_d, giant_model_name, 'common'][testing, :] = np.squeeze(np.array(pred_Y_comm))

    h5file.close()
    print 'H5 File Done: ', hdf_filename

    ### ALSO SAVE MODEL_DATA: 
    pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'wb'))

### Generalization of neural dynamics across task -- model 7###
def model_individual_cell_tuning_curves_co_obs_spec_dyn(hdf_filename='_models_to_pred_mn_diffs_co_obs_spec', 
    animal='grom', n_folds = 5, model_set_number = 7, ndays = None, history_bins_max = 1,
    ridge = True, use_action = False):

    if use_action:
        models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']
    else:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']                          
    
    predict_key = 'spks'

    if animal == 'jeev':
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
    else:
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

    if hdf_filename is None: 
        x = datetime.datetime.now()
        hdf_filename = animal + '_' + x.strftime("%Y_%m_%d_%H_%M_%S") + '.h5'
        hdf_filename = pref + hdf_filename; 
    else:
        hdf_filename = pref + animal + hdf_filename + '_model_set%d.h5' %model_set_number

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    ridge_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    if animal == 'grom':
        order_dict = co_obs_tuning_matrices.ordered_input_type
        input_type = analysis_config.data_params['grom_input_type']

    elif animal == 'jeev':
        order_dict = file_key.ordered_task_filelist
        input_type = file_key.task_filelist

    if ndays is None:
        pass
    else:
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, spikes, sub_spk_temp_all, sub_push_all = get_spike_kinematics(animal, day, 
            order_dict[i_d], history_bins_max)

        ### Replicate datastructure for saving later: 
        ### Want to save neural push, task, target 
        model_data[i_d, 'spikes'] = spikes.copy();
        model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))
        model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
        model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))
        model_data[i_d, 'bin_num'] = np.squeeze(np.array(data_temp['bin_num']))
        
        ### Models -- save predicitons
        for mod in models_to_include:

            if predict_key == 'spks':
                model_data[i_d, mod] = np.zeros((spikes.shape[0], spikes.shape[1], 3)) 
                
            elif predict_key == 'psh':
                model_data[i_d, mod] = np.zeros_like(model_data[i_d, 'np']) 
                #model_data[i_d, mod, 'common'] = np.zeros_like(model_data[i_d, 'np'])

        ##########################################################################
        #### Fit all neurons simulataneously, and add many more model options ###
        #########################################################################
        ### Get training and testing datasets -- 
        train_ix = dict();
        test_wi_ix = dict(); 
        test_x_ix = dict(); 

        nneur = sub_spk_temp_all.shape[2]
        T_all = len(model_data[i_d,'task'])

        i_trg = -1; 

        ix_all0 = np.nonzero( model_data[i_d, 'task'] == 0 )[0]
        ix_all1 = np.nonzero( model_data[i_d, 'task'] == 1 )[0]

        print ''
        print 'fitting on the max amt of data from task w least data -- i.e. models arent diff bc diff amts of data'
        print ''

        max_data_train = int(np.floor((1. - (1./n_folds))*np.min([len(ix_all0), len(ix_all1)])))

        ### Also fit on both at the end: 
        for i_t in range(3):
            
            ### Get all indices that coudl be used to train task/target; 
            #ix_all = (model_data[i_d, 'task'] == i_t) & (model_data[i_d, 'trg'] == i_trg); 
            

            ### For each task, select 20% indices to train, 20 % indicies in other task to test; 
            if i_t in [0, 1]:
                ix_all = np.nonzero( model_data[i_d, 'task'] == i_t )[0]
                not_i_t = np.mod(i_t + 1, 2)

                ### Shuffle all this; 
                ix_all = ix_all[np.random.permutation(len(ix_all))]
                ntmp = float(len(ix_all))

                ix_not_i_t = np.nonzero(model_data[i_d, 'task'] == not_i_t)[0]
                ix_not_i_t = ix_not_i_t[np.random.permutation(len(ix_not_i_t))]
                ntmp_not_it = float(len(ix_not_i_t))

                # ix_all, ix_not_i_t
                

            else: ### Combined tasks: 
                ix_all0 = np.nonzero( model_data[i_d, 'task'] == 0)[0]
                ix_all1 = np.nonzero( model_data[i_d, 'task'] == 1)[0]

                ix_all0 = ix_all0[np.random.permutation(len(ix_all0))]
                ix_all1 = ix_all1[np.random.permutation(len(ix_all1))]

                ntmp0 = len(ix_all0)
                ntmp1 = len(ix_all1)
                ntmp  = len(model_data[i_d, 'task'])

                # ix_all0, ix_all1;


            ### Split this up into train / test; 
            for i_f, fold_perc in enumerate(np.arange(0, 1., 1./n_folds)):
                
                if i_t in [0, 1]:

                    ### Within task testing ### -- task specific testing: 
                    tst = ix_all[int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]

                    ### Within task training ###
                    trn0 = np.array([j for i, j in enumerate(ix_all) if j not in tst])
                    
                    ################################################
                    ##### Manuever to match samples for CO / OBS ###
                    ################################################
                    
                    trn0 = trn0[np.random.permutation(len(trn0))].copy() ### Randomize train
                    
                    ### Only get top train data
                    trn = trn0[:max_data_train]
                    tst = np.array([j for i, j in enumerate(ix_all) if j not in trn])

                    ### Across task testing ###
                    test_x = ix_not_i_t[int(fold_perc*ntmp_not_it):int((fold_perc+(1./n_folds))*ntmp_not_it)]

                    test_wi_ix[i_t, i_trg, i_f] = tst; ### Test within task 
                    train_ix[i_t, i_trg, i_f] = trn; ### Train task
                    test_x_ix[i_t, i_trg, i_f] = test_x; ### Test x task

                    assert(len(tst) + len(trn) == ntmp)

                else: 
                    ### Get training from both task: 
                    tst0 = ix_all0[int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
                    tst1 = ix_all1[int(fold_perc*ntmp):int((fold_perc+(1./n_folds))*ntmp)]
                    tst = np.hstack((tst0, tst1))
                    trn = np.array([i for i in range(int(ntmp)) if i not in tst])

                    ################################################
                    ##### Manuever to match samples for CO / OBS ###
                    ################################################
                    trn = trn[np.random.permutation(len(trn))] ### Randomize train
                    ### Only get top train data
                    trn = trn[:int(max_data_train)]
                    tst = np.array([j for j in range(int(ntmp)) if j not in trn])

                    test_wi_ix[i_t, i_trg, i_f] = tst; ### Test within task 
                    train_ix[i_t, i_trg, i_f] = trn; ### Train task
                    test_x_ix[i_t, i_trg, i_f] = []; ### Test x task

            ### Clear how to split data: 
            print ("Task %d, Day %d, NPts: %d" %(i_t, i_d, len(train_ix[i_t, i_trg, 0])))

            for i_fold in range(n_folds):

                ### TEST DATA ####
                testing = test_wi_ix[i_t, i_trg, i_fold]
                data_temp_dict_test = panda_to_dict(data_temp.iloc[testing])
                data_temp_dict_test['spks'] = spikes[testing]
                data_temp_dict_test['pshx'] = sub_push_all[testing, 0]
                data_temp_dict_test['pshy'] = sub_push_all[testing, 1]
                data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

                ### TRAIN DATA #### -- training for the task-specific. 
                training = train_ix[i_t, i_trg, i_fold]
                data_temp_dict = panda_to_dict(data_temp.iloc[training])
                data_temp_dict['spks'] = spikes[training]
                data_temp_dict['pshx'] = sub_push_all[training, 0]
                data_temp_dict['pshy'] = sub_push_all[training, 1]
                data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))

                ### TRAIN DATA #### -- training for the GENERAL neural dynamics
                test_x = test_x_ix[i_t, i_trg, i_fold]
                if len(test_x) > 0:
                    data_temp_dict_x = panda_to_dict(data_temp.iloc[test_x])
                    data_temp_dict_x['spks'] = spikes[test_x]
                    data_temp_dict_x['pshx'] = sub_push_all[test_x, 0]
                    data_temp_dict_x['pshy'] = sub_push_all[test_x, 1]
                    data_temp_dict_x['psh'] = np.hstack(( data_temp_dict_x['pshx'], data_temp_dict_x['pshy']))
                else:
                    data_temp_dict_x = None; 

                assert(len(testing) + len(training) == ntmp)
                
                ##############################################
                ################# RUNNING ALL THE MODELS ###
                ##############################################
                # Velocity models: number of lags:
                ### NEW  ###
                model_var_list = [[np.array([-1]), 'hist_1']]

                ## Go through the models
                models_added = []; 
                for im, (model_vars, model_nm) in enumerate(model_var_list):
                    
                    ### State tuning, velocity: 
                    for add_state in range(1):
                        
                        if add_state == 1:
                            #vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                            print('Removing velocity')
                            vel_model_nms = []; 
                            pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                
                        else:
                            vel_model_nms = pos_model_nms = []; 

                        ### Use push: 
                        for add_push in range(2): 
                            if add_push == 0:
                                push_model_nms = []; 

                            elif add_push == 1:

                                ### ONLY ADDING CURRENT PUSH
                                push_model_nms, push_model_str = lag_ix_2_var_nm(model_vars, 'psh', 
                                    include_action_lags=False)  

                            ### Get neural lags
                            for add_neur in range(1, 2):
                                if add_neur == 1:
                                    neur_nms0, _ = lag_ix_2_var_nm(model_vars, 'neur', nneur, neur_lag=[-1])
                                else:
                                    neur_nms0 = []; 

                                variables = np.hstack((vel_model_nms, pos_model_nms, push_model_nms,
                                            neur_nms0))
                                
                                if ridge:
                                    giant_model_name = model_nm + 'pos_'+str(add_state) + 'psh_'+str(add_push) +'spksm_'+str(add_neur)+'_spksp_0'
                                    save_model = False

                                    if giant_model_name in models_to_include:

                                        alpha = ridge_dict[animal][0][i_d, giant_model_name]

                                        #print('Using alpha %.2f, adding model %s' %(alpha, giant_model_name))
                                        
                                        model_2, pred_Y, pred2 = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, 
                                            test_data=data_temp_dict_test, test_data2=data_temp_dict_x, alpha = alpha)

                                        ### Add the giant mdoel name. 
                                        models_added.append(giant_model_name)
                                        save_model = True; 
                                
                                else:
                                    raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                                    model_ = ols(st, data_temp_dict).fit()

                                    ### Important --- name the model: 
                                    #name = model_nm + 'pos_'+str(add_pos) + 'psh_'+str(add_push) +'spks_'+str(add_neural)
                                    
                                if save_model:
                                    ### Save models, make predictions ####
                                    ### Need to figure out which spikes are where:

                                    ### Save for this particular task comparison: 
                                    model_data[i_d, giant_model_name][testing, :, i_t] = np.squeeze(np.array(pred_Y))

                                    if len(test_x) > 0:
                                        model_data[i_d, giant_model_name][test_x, :, i_t] = np.squeeze(np.array(pred2))

    h5file.close()
    print 'H5 File Done: ', hdf_filename

    ### ALSO SAVE MODEL_DATA: 
    pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, use_action), 'wb'))

### Generalization of neural dynamics across task / target -- model 7###
def model_individual_cell_tuning_curves_co_obs_tsk_tg_dyn(hdf_filename='_models_to_pred_mn_diffs_co_obs_tsk_tg_spec', 
    animal='grom', n_folds = 5, model_set_number = 7, ndays = 1, history_bins_max = 1,
    ridge = True, use_action = False):
    
    eigenvalues = dict()

    if use_action:
        models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']# s_{t-1}, y_{t-1}, a_t                            
    else:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']# s_{t-1}, y_{t-1}, a_t                            
    
    predict_key= 'spks'

    if animal == 'jeev':
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/'
    else:
        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'

    if hdf_filename is None: 
        x = datetime.datetime.now()
        hdf_filename = animal + '_' + x.strftime("%Y_%m_%d_%H_%M_%S") + '.h5'
        hdf_filename = pref + hdf_filename; 
    else:
        hdf_filename = pref + animal + hdf_filename + '_model_set%d.h5' %model_set_number

    ### Place to save models: 
    model_data = dict(); 

    ### Get the ridge dict: 
    ridge_dict = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/max_alphas_ridge_model_set%d.pkl' %model_set_number, 'rb')); 

    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, tables.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass #

    tuning_dict = {}
    if animal == 'grom':
        order_dict = co_obs_tuning_matrices.ordered_input_type
        input_type = analysis_config.data_params['grom_input_type']

    elif animal == 'jeev':
        order_dict = file_key.ordered_task_filelist
        input_type = file_key.task_filelist

    if ndays is None:
        pass
    else:
        order_dict = [order_dict[i] for i in range(ndays)]
        input_type = [input_type[i] for i in range(ndays)]

    h5file = tables.openFile(hdf_filename, mode="w", title=animal+'_tuning')

    for i_d, day in enumerate(input_type):
        
        # Get spike data from data fcn
        data, data_temp, spikes, sub_spk_temp_all, sub_push_all = get_spike_kinematics(animal, day, 
            order_dict[i_d], history_bins_max)

        ### Replicate datastructure for saving later: 
        ### Want to save neural push, task, target 
        model_data[i_d, 'spikes'] = spikes.copy();
        model_data[i_d, 'task'] = np.squeeze(np.array(data_temp['tsk']))
        model_data[i_d, 'trg'] = np.squeeze(np.array(data_temp['trg']))
        model_data[i_d, 'np'] = np.squeeze(np.array(sub_push_all))
        model_data[i_d, 'bin_num'] = np.squeeze(np.array(data_temp['bin_num']))

        ### Models -- save predictions
        for mod in models_to_include:

            if predict_key == 'spks':
                model_data[i_d, mod] = np.zeros((spikes.shape[0], spikes.shape[1], 16)) + np.nan
            # elif predict_key == 'psh':
            #     model_data[i_d, mod] = np.zeros_like(model_data[i_d, 'np']) 
            #     #model_data[i_d, mod, 'common'] = np.zeros_like(model_data[i_d, 'np'])

        ##########################################################################
        #### Fit all neurons simulataneously, and add many more model options ###
        #########################################################################
        ### Get training and testing datasets -- 
        train_ix = dict();
        test_wi_ix = dict(); 
        test_x_ix = dict(); 

        nneur = sub_spk_temp_all.shape[2]
        T_all = len(model_data[i_d,'task'])

        i_trg = -1; 

        max_data_train = 10e5; 
        for i_t in range(2):
            for i_trg in range(8):
                ix = (model_data[i_d, 'task'] == i_t) & (model_data[i_d, 'trg'] == i_trg)
                max_data_train = np.min([max_data_train, np.sum(ix)])

        max_data_train = int(max_data_train)
        N = len(model_data[i_d, 'task'])

        for i_t in range(2):
            for i_trg in range(8): 

                index_fold = i_t*8 + i_trg; 

                print ("Task %d" %(i_t))

                ### Get all indices that coudl be used to train task/target;
                ix_all = np.nonzero( (model_data[i_d, 'task'] == i_t) & (model_data[i_d, 'trg'] == i_trg) )[0]
                
                ### Train on all of this -- unless its max is higher than the max expected: 
                ### Shuffle all this; 
                ix_all = ix_all[np.random.permutation(len(ix_all))]
                training = ix_all[:max_data_train]

                ### Add everything else
                testing = np.array([j for j in range(N) if j not in training])

                ### Test: 
                data_temp_dict_test = panda_to_dict(data_temp.iloc[testing])
                data_temp_dict_test['spks'] = spikes[testing]
                data_temp_dict_test['pshx'] = sub_push_all[testing, 0]
                data_temp_dict_test['pshy'] = sub_push_all[testing, 1]
                data_temp_dict_test['psh'] = np.hstack(( data_temp_dict_test['pshx'], data_temp_dict_test['pshy']))

                ### TRAIN DATA #### -- training for the task-specific. 
                data_temp_dict = panda_to_dict(data_temp.iloc[training])
                data_temp_dict['spks'] = spikes[training]
                data_temp_dict['pshx'] = sub_push_all[training, 0]
                data_temp_dict['pshy'] = sub_push_all[training, 1]
                data_temp_dict['psh'] = np.hstack(( data_temp_dict['pshx'], data_temp_dict['pshy']))

                assert(len(testing) + len(training) == N)
                
                ##############################################
                ################# RUNNING ALL THE MODELS ###
                ##############################################
                model_var_list = [[np.array([-1]), 'hist_1']]

                ## Go through the models
                models_added = []; 
                for im, (model_vars, model_nm) in enumerate(model_var_list):
                    
                    ### State tuning, velocity: 
                    for add_state in range(1):
                        
                        if add_state == 1:
                            #vel_model_nms, _ = lag_ix_2_var_nm(model_vars, 'vel')
                            print('Removing velocity')
                            vel_model_nms = []; 
                            pos_model_nms, _ = lag_ix_2_var_nm(model_vars, 'pos')
                
                        else:
                            vel_model_nms = pos_model_nms = []; 

                        ### Use push: 
                        for add_push in range(2): 
                            if add_push == 0:
                                push_model_nms = []; 

                            elif add_push == 1:

                                ### ONLY ADDING CURRENT PUSH
                                push_model_nms, push_model_str = lag_ix_2_var_nm(model_vars, 'psh', 
                                    include_action_lags=False)  

                            ### Get neural lags
                            for add_neur in range(1, 2):
                                if add_neur == 1:
                                    neur_nms0, _ = lag_ix_2_var_nm(model_vars, 'neur', nneur, neur_lag=[-1])
                                else:
                                    neur_nms0 = []; 

                                variables = np.hstack((vel_model_nms, pos_model_nms, push_model_nms,
                                            neur_nms0))
                                
                                if ridge:
                                    giant_model_name = model_nm + 'pos_'+str(add_state) + 'psh_'+str(add_push) +'spksm_'+str(add_neur)+'_spksp_0'
                                    save_model = False

                                    if giant_model_name in models_to_include:

                                        alpha = ridge_dict[animal][0][i_d, giant_model_name]

                                        print('Using alpha %.2f, adding model %s' %(alpha, giant_model_name))
                                        
                                        model_2, pred_Y, pred2 = fit_ridge(data_temp_dict[predict_key], data_temp_dict, variables, 
                                            test_data=data_temp_dict_test, test_data2=None, alpha = alpha)

                                        ### Add the giant mdoel name. 
                                        models_added.append(giant_model_name)
                                        save_model = True; 
                                
                                else:
                                    raise Exception('Need to figure out teh stirng business again -- removed for clarity')
                                    model_ = ols(st, data_temp_dict).fit()

                                    ### Important --- name the model: 
                                    #name = model_nm + 'pos_'+str(add_pos) + 'psh_'+str(add_push) +'spks_'+str(add_neural)
                                    
                                if save_model:
                                    ### Save models, make predictions ####
                                    ### Need to figure out which spikes are where:
                                    ### Save for this particular task comparison: 
                                    model_data[i_d, giant_model_name][testing, :, index_fold] = np.squeeze(np.array(pred_Y))
                                    ev, _ = np.linalg.eig(model_2.coef_)
                                    eigenvalues[i_d, i_t, i_trg] = ev;  


    h5file.close()
    print 'H5 File Done: ', hdf_filename

    ### ALSO SAVE MODEL_DATA: 
    pickle.dump(model_data, open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_tsk_tg_use_action%s.pkl' %(model_set_number, use_action), 'wb'))
    return eigenvalues

def tmp_run_jeev():

    for use_action in [True, False]:
        model_individual_cell_tuning_curves_co_obs_spec_dyn(animal='jeev', 
            model_set_number = 7, ndays = 1, use_action = use_action)

        model_individual_cell_tuning_curves_co_obs_tsk_tg_dyn(animal='jeev', 
            model_set_number = 7, ndays = 1, use_action = use_action)

def plot_eigs(ev):
    ### for each day: 
    i_d = 0; 

    f, ax = plt.subplots()

    color = ['b', 'r']


    for t in range(2):
        for trg in range(8):

            evi = ev[(i_d, t, trg)]

            ### only take top 80% EVs: 
            evi_ix = np.nonzero( (np.abs(evi)/np.sum(np.abs(evi))) > 0.8)[0]

            angs = np.angle(evi)

            hz = np.abs(angs)/(2*np.pi*.1)
            tau = -1./np.log(np.abs(evi))*.1 # Ti

            ax.plot(tau, hz, '.', color = color[t])

def plot_all_mn_diff_plots():
    for animal in ['grom', 'jeev']:
        for _, (null, pot) in enumerate(zip([True, False, False], [False, True, False])):
            print 'Animal -- ', animal, ' Null -- ', null, ' Pot -- ', pot
            mean_diffs_plot(animal = animal, only_null = null, only_potent = pot)

### Use model predictions to generate means -- potent and null options included. 
def mean_diffs_plot(animal = 'grom', min_obs = 15, load_file = 'default', dt = 1, 
    important_neurons = True, skip_ind_plots = True, only_null = False, only_potent = False, 
    model_set_number = 3, ndays = None, next_pt_pred = True, plot_pred_vals = True,):

    '''
    next_pt_pred -- instead of quantifying how models predict CURRENT time point (aligned to action bin),
        quantify instead how well modesl predict NEXT time point 

    plot_pred_vals -- for model plot 
        a) true neur vs. pred neur, 
        b) tru KG vs. pred KG, 
        c) task neur diff vs. pred task neur diff,
        d) task diff KG, vs. pred task diff KG, d) 
    '''
    
    savedir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

    ### Magnitude boundaries: 
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    marks = ['-', '--']

    if important_neurons:
        if animal == 'grom':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))
        
        elif animal == 'jeev':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))

    ### List the models to analyze
    if model_set_number == 1:
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', # only action, no state
                             'prespos_1psh_1spksm_0_spksp_0', # Current state, action, no spks
                             'hist_fut_4pos_1psh_1spksm_0_spksp_0', # State/action and lags, no spks
                             'hist_fut_4pos_1psh_1spksm_1_spksp_0', # State/action and lags, only spks_{t-1}
                             'hist_fut_4pos_1psh_1spksm_1_spksp_1']
    
    elif model_set_number in [3]:
        models_to_include = ['prespos_0psh_0spksm_1_spksp_0',
                             'prespos_0psh_0spksm_1_spksp_0potent',
                             'prespos_0psh_0spksm_1_spksp_0null',
                             'hist_1pos_0psh_0spksm_1_spksp_0',
                             'hist_1pos_0psh_0spksm_1_spksp_0potent',
                             'hist_1pos_0psh_0spksm_1_spksp_0null']
    elif model_set_number in [2]:
        models_to_include = ['prespos_0psh_0spksm_1_spksp_0',
        'hist_1pos_0psh_0spksm_1_spksp_0',
        'hist_1pos_0psh_1spksm_0_spksp_0']

    if model_set_number in [1, 3]:
        key = 'spks'

    elif model_set_number in [2]:
        key = 'np'

    if load_file is None:
        ### get the model predictions: 
        model_individual_cell_tuning_curves(animal=animal, history_bins_max=4, 
            ridge=True, include_action_lags = True, return_models = True, 
            models_to_include = models_to_include)
    
    elif load_file == 'default':
        ### load the models: 
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
        
    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    hist_bins = np.arange(15)

    ### Pooled over all days: 
    f_all, ax_all = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (len(models_to_include)*3, 6))
    
    TD_all = dict();
    PD_all = dict(); 

    for mod in models_to_include:
        for sig in ['all', 'sig']:
            TD_all[mod, sig] = []; 
            PD_all[mod, sig] = []; 
        
    for i_d in range(ndays[animal]):

        ### Basics -- get the binning for the neural push commands: 
        neural_push = model_dict[i_d, 'np']

        ### Commands
        commands = subspace_overlap.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
        
        ### Get task / target
        tsk = model_dict[i_d, 'task']
        targ = model_dict[i_d, 'trg']
        bin_num = model_dict[i_d, 'bin_num']

        ### Now go through each task targ and assess when there are enough observations: 
        y_true = model_dict[i_d, key]

        T, N = y_true.shape

        if important_neurons:
            important_neur = imp_file[(i_d, animal, 'svd')]
        else:
            important_neur = np.arange(N); 

        if skip_ind_plots:
            important_neur = []; 

        f_dict = dict(); 

        ### Get the decoder 
        ### Get the decoder ###
        if animal == 'grom':
            KG, KG_null, KG_pot = get_KG_decoder_grom(i_d)

        if animal == 'jeev':
            KG, KG_null, KG_pot = get_KG_decoder_jeev(i_d)

        if np.logical_or(only_null, only_potent):
            if np.logical_and(only_null, only_potent):
                raise Exception('Cant have null and potent')
            else:
                y_true = get_decomp_y(KG_null, KG_pot, y_true, only_null = only_null, only_potent=only_potent)

        ### Only plot distribution diffs for important neurons: 
        for n in important_neur:
            f, ax = plt.subplots(ncols=8, nrows = 4, figsize = (12, 6))
            f_dict[n] = [f, ax]; 
        print('Important neruons %d' %len(important_neur))
        
        ### Make the spiking histograms: 
        sig_diff = np.zeros((N, 4, 8))
        index_dictionary = {}

        for i_mag in range(4):
            for i_ang in range(8):

                sig_i_t = 0; 

                for i_t in range(2):
                    ### Select by task / target / mag / ang; 
                    #ix = (neural_push[:,0] == i_mag) & (neural_push[:,1] == i_ang) & (targ == i_trg) & (tsk == i_t)
                    ix = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == i_t)

                    if np.sum(ix) >= min_obs:
                        print 'Plotting Day %d, Task %d, Mag %d Ang %d' %(i_d, i_t, i_mag, i_ang)
                        sig_i_t += 1
                        
                        ### Plotting: 
                        y_obs = y_true[ix, :]

                        ## Which indices: 
                        I = i_mag*8 + i_ang

                        ### Indexing: 
                        for n in range(N):

                            ### Plot important neurons; 
                            if n in important_neur:

                                ## get the axis
                                axi = f_dict[n][1][i_mag, i_ang]

                                ### histogram
                                h, biz = np.histogram(y_obs[:, n], hist_bins)

                                ### Plot line
                                axi.plot(biz[:-1] + .5*dt, h / float(np.sum(h)), '-',
                                    color = cmap_list[i_t])

                                ## Plot the mean; 
                                axi.vlines(np.mean(y_obs[:, n]), 0, .5, cmap_list[i_t])

                # for each neuron figure if enough observations of the i_mag / i_ang to plot; 
                if sig_i_t == 2:

                    if next_pt_pred:
                        print 'Aligning to mag %d, ang %d, next step ahead'
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0)
                        ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1)

                        ix0 = np.nonzero(ix0 == True)[0]
                        ix0 = ix0 + 1; 
                        ix0 = ix0[ix0 < len(tsk)]
                        ix0 = ix0[bin_num[ix0] > 0]

                        ix1 = np.nonzero(ix1 == True)[0]
                        ix1 = ix1 + 1; 
                        ix1 = ix1[ix1 < len(tsk)]
                        ix1 = ix1[bin_num[ix1] > 0]

                    else:
                        ## Find relevant commands: 
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0)
                        ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1)
                    
                        ix0 = np.nonzero(ix0 == True)[0]
                        ix1 = np.nonzero(ix1 == True)[0]

                    index_dictionary[i_mag, i_ang] = [ix0, ix1]

                    for n in range(N):
                        ### Two-sided test for distribution difference. If pv > 0.05 can't reject hypothesis that 
                        ### two samples come from the same distribution. 
                        _, pv = scipy.stats.ks_2samp(y_true[ix0, n], y_true[ix1, n])

                        sig_diff[n, i_mag, i_ang] = pv < 0.05; 

                        if n in important_neur:

                            axi = f_dict[n][1][i_mag, i_ang]
                            if pv < 0.05:
                                axi.set_title('Mag %d, Ang %d ***, \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                    fontsize=8)
                            else:
                                axi.set_title('Mag %d, Ang %d n.s., \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                    fontsize=8)
                    else:
                        if n in important_neur:
                            axi.set_title('Mag %d, Ang %d n.s., \nCO N %d, OBS N %d' %(i_mag, i_ang, np.sum(ix0), np.sum(ix1)),
                                fontsize=8)

        for n in important_neur:
            f_dict[n][0].tight_layout()
            f_dict[n][0].savefig(savedir + animal + '_day_' + str(i_d) + '_n_'+str(n) + '.png')
            plt.close(f_dict[n][0])

        ###################################
        ### Now get the diff models: ######
        print 'Done with day -- -1 --'
        sig = ['k', 'r']
        f, ax = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (3*len(models_to_include), 6))

        ### Rows are CO / OBS; 
        f_, ax_ = plt.subplots(ncols = len(models_to_include), nrows = 4, figsize = (3*len(models_to_include), 8))

        for i_m, mod in enumerate(models_to_include):

            #### These are using the null/potent 
            if 'potent' in mod:
                modi = mod[:-6]
                y_pred = model_dict[i_d, modi, 'pot']
                #y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = False, only_potent=True)
            
            elif 'null' in mod:
                modi = mod[:-4]
                y_pred = model_dict[i_d, modi, 'null']
                #y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = False, only_potent=True)                
            else:
                y_pred = model_dict[i_d, mod]

            # ### Get null activity ###
            # if np.logical_or(only_null, only_potent):
            #     y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = only_null, only_potent=only_potent)

            ax[0, i_m].set_title(mod, fontsize=6)
            TD = []; PD = []; TD_s = []; PD_s = [];

            pred = []; obs = []; 
            predkg = []; obskg = [];
            diff_tru = []; diff_trukg = []; 
            diff_pred = []; diff_predkg = []; 

            ### Make a plot
            for i_mag in range(4):
                for i_ang in range(8):

                    if tuple([i_mag, i_ang]) in index_dictionary.keys():
                        ix0, ix1 = index_dictionary[i_mag, i_ang]

                        assert np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs)

                        if key == 'spks':
                            #### True difference over all neurons -- CO vs. OBS:
                            tru_co = np.mean(y_true[ix0, :], axis=0)
                            tru_ob = np.mean(y_true[ix1, :], axis=0)

                            pred_co = np.mean(y_pred[ix0, :], axis=0)
                            pred_ob = np.mean(y_pred[ix1, :], axis=0)

                            tru_diff = tru_co - tru_ob
                            pred_diff = pred_co - pred_ob

                        elif key == 'np':
                            ### Mean angle: 
                            # mean_co = math.atan2(np.mean(y_true[ix0, 1]), np.mean(y_true[ix0, 0]))
                            # mean_ob = math.atan2(np.mean(y_true[ix1, 1]), np.mean(y_true[ix1, 0]))
                            
                            # pred_mean_co = math.atan2(sw.mean(y_pred[ix0, 1]), np.mean(y_pred[ix0, 0]))
                            # pred_mean_ob = math.atan2(np.mean(y_pred[ix1, 1]), np.mean(y_pred[ix1, 0]))
                            
                            # ### do an angular difference: 
                            # tru_diff = ang_difference(np.array([mean_co]), np.array([mean_ob]))
                            # pred_diff = ang_difference(np.array([pred_mean_co]), np.array([pred_mean_ob]))

                            tru_co = np.mean(y_true[ix0, :], axis=0)
                            tru_ob = np.mean(y_true[ix1, :], axis=0)

                            pred_co = np.mean(y_pred[ix0, :], axis=0)
                            pred_ob = np.mean(y_pred[ix1, :], axis=0)

                            tru_diff = tru_co - tru_ob
                            pred_diff = pred_co - pred_ob

                        for n, (td, pd) in enumerate(zip(tru_diff, pred_diff)):
                            if sig_diff[n, i_mag, i_ang] == 1:
                                ax[1, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)
                                ax[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)

                                ax_all[1, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)
                                ax_all[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=1.)


                                TD_s.append(td);
                                PD_s.append(pd);

                                TD_all[mod, 'sig'].append(td);
                                PD_all[mod, 'sig'].append(pd);

                            ax[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=0.2)
                            ax_all[0, i_m].plot(td, pd, '.', color = sig[int(sig_diff[n, i_mag, i_ang])], alpha=0.2)

                            TD.append(td);
                            PD.append(pd);
                            TD_all[mod, 'all'].append(td);
                            PD_all[mod, 'all'].append(pd);

                        if plot_pred_vals:
                            ax_[0, i_m].plot(tru_co, pred_co, 'b.')
                            ax_[0, i_m].plot(tru_ob, pred_ob, 'r.')
                            ax_[2, i_m].plot(tru_co - tru_ob, pred_co - pred_ob, 'k.')

                            #### Split by X / Y 
                            if key != 'np':
                                ax_[1, i_m].plot(np.dot(KG, tru_co.T)[0], np.dot(KG, pred_co.T)[0], 'b.')
                                ax_[1, i_m].plot(np.dot(KG, tru_co.T)[1], np.dot(KG, pred_co.T)[1], 'b.', alpha = .3)

                                ax_[1, i_m].plot(np.dot(KG, tru_ob.T)[0], np.dot(KG, pred_ob.T)[0], 'r.')
                                ax_[1, i_m].plot(np.dot(KG, tru_ob.T)[1], np.dot(KG, pred_ob.T)[1], 'r.', alpha = .3)
                            
                                ###### PLOT DIFFERENCE ########
                                ax_[3, i_m].plot(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T), np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T),'k.')  
                            
                            diff_tru.append(tru_co - tru_ob)
                            diff_pred.append(pred_co - pred_ob)

                            if key != 'np':
                                diff_trukg.append(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T))
                                diff_predkg.append(np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T))

                            pred.append(pred_co)
                            pred.append(pred_ob)
                            obs.append(tru_co)
                            obs.append(tru_ob)

                            if key != 'np':
                                predkg.append(np.dot(KG, pred_co.T))
                                predkg.append(np.dot(KG, pred_ob.T))

                                obskg.append(np.dot(KG, tru_co.T))
                                obskg.append(np.dot(KG, tru_ob.T))

                            cokg = np.mean(neural_push[ix0, :], axis=0)
                            obkg = np.mean(neural_push[ix1, :], axis=0)

                            if key != 'np':
                                assert np.sum(np.abs(cokg - np.dot(KG, tru_co.T))) < 5e5
                                assert np.sum(np.abs(obkg - np.dot(KG, tru_ob.T))) < 5e5

                            ax_[0, i_m].set_title('True Y_t vs. Pred Y_t\nModel %s' %mod)
                            ax_[1, i_m].set_title('True K*Y_t vs. Pred K*Y_t\nModel %s' %mod)

            if plot_pred_vals:
                pred = np.hstack((pred)).reshape(-1)
                obs = np.hstack((obs)).reshape(-1)

                if key != 'np':
                    predkg = np.hstack((predkg)).reshape(-1)
                    obskg = np.hstack((obskg)).reshape(-1)
                
                diff_tru = np.hstack((diff_tru)).reshape(-1)
                diff_pred = np.hstack((diff_pred)).reshape(-1)

                if key != 'np':
                    diff_trukg = np.hstack((diff_trukg)).reshape(-1)
                    diff_predkg = np.hstack((diff_predkg)).reshape(-1)

                slp,intc,rv,pv,err = scipy.stats.linregress(obs, pred)
                x_ = np.linspace(np.min(obs), np.max(obs), 100)
                y_ = slp*x_ + intc; 
                ax_[0, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                ax_[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                #ax_[0, i_m].set_ylim([0, 9])
                
                slp,intc,rv,pv,err = scipy.stats.linregress(diff_tru, diff_pred)
                x_ = np.linspace(np.min(diff_tru), np.max(diff_tru), 100)
                y_ = slp*x_ + intc; 
                ax_[2, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                ax_[2, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                #ax_[2, i_m].set_ylim([-2, 2])

                if key != 'np':
                    slp,intc,rv,pv,err = scipy.stats.linregress(obskg, predkg)
                    x_ = np.linspace(np.min(obskg), np.max(obskg), 100)
                    y_ = slp*x_ + intc; 
                    ax_[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                    ax_[1, i_m].set_ylim([-2, 2])
                
                    slp,intc,rv,pv,err = scipy.stats.linregress(diff_trukg, diff_predkg)
                    x_ = np.linspace(np.min(diff_trukg), np.max(diff_trukg), 100)
                    y_ = slp*x_ + intc; 
                    ax_[3, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[3, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
                    ax_[3, i_m].set_ylim([-.5, .5])

            ### Lets do a linear correlation: 
            slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD)), np.hstack((PD)))
            x_ = np.linspace(np.min(np.hstack((TD))), np.max(np.hstack((TD))), 100)
            y_ = slp*x_ + intc; 
            ax[0, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            ax[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))

            slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD_s)), np.hstack((PD_s)))
            x_ = np.linspace(np.min(np.hstack((TD_s))), np.max(np.hstack((TD_s))), 100)
            y_ = slp*x_ + intc; 
            ax[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            ax[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
            print('Done with day -- %d --'%i_m)
        
        f.tight_layout()
        f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'.png')

        print 'Done with day -- end --'

    ### Get all the stuff for the ax_all; 
    for i_m, mod in enumerate(models_to_include):
        for sig, sig_nm in enumerate(['all', 'sig']):
            x = np.hstack((TD_all[mod, sig_nm]))
            y = np.hstack((PD_all[mod, sig_nm]))

            slp,intc,rv,pv,err = scipy.stats.linregress(x, y)
            x_ = np.linspace(np.min(x), np.max(x), 100)
            y_ = slp*x_ + intc; 
            ax_all[sig, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            ax_all[sig, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.4f \npv = %.4f' %(rv, pv))
    f_all.tight_layout()
    f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'.png')

def mean_diffs_plot_x_condition(animal = 'grom', min_obs = 15, load_file = 'default', 
    important_neurons = True, skip_ind_plots = True, only_null = False, only_potent = False, 
    model_set_number = 8, ndays = 1, next_pt_pred = True, plot_pred_vals = True,
    within_task = False, within_task_task = 0, use_action = False, restrict_radius = 1000,
    n_steps_prop = 1, cov = False, kg_diff_angle = False):

    '''
    same as mean_diffs_plot, now taking mean differences across condition 
        - only do across conditions that are across task; 
    search for commands that are the SAME. 
        - predict their NEXT time step. 
        - what is the mean diff in their NEXT time step vs. predicted diff in NEXT time step
    '''
    
    savedir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

    ### Magnitude boundaries: 
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    marks = ['-', '--']

    if important_neurons:
        if animal == 'grom':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))
        
        elif animal == 'jeev':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))

    ### List the models to analyze
    if model_set_number == 1:
        # models_to_include = ['prespos_0psh_1spksm_0_spksp_0', # only action, no state
        #                      'prespos_1psh_1spksm_0_spksp_0', # Current state, action, no spks
        #                      'hist_fut_4pos_1psh_1spksm_0_spksp_0', # State/action and lags, no spks
        #                      'hist_fut_4pos_1psh_1spksm_1_spksp_0', # State/action and lags, only spks_{t-1}
        #                      'hist_fut_4pos_1psh_1spksm_1_spksp_1']
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', # action only 
                            'hist_1pos_3psh_1spksm_0_spksp_0', # all state variables
                            'hist_1pos_3psh_1spksm_1_spksp_0'] # all state variables plus y_{t-1}

    elif model_set_number in [3]:
        models_to_include = ['prespos_0psh_0spksm_1_spksp_0',
                             'prespos_0psh_0spksm_1_spksp_0potent',
                             'prespos_0psh_0spksm_1_spksp_0null',
                             'hist_1pos_0psh_0spksm_1_spksp_0',
                             #'hist_1pos_0psh_1spksm_0_spksp_0']
                             'hist_1pos_0psh_0spksm_1_spksp_0potent',
                             'hist_1pos_0psh_0spksm_1_spksp_0null']

    elif model_set_number in [2]:
        models_to_include = ['prespos_0psh_0spksm_1_spksp_0',
        'hist_1pos_0psh_0spksm_1_spksp_0',
        'hist_1pos_0psh_1spksm_0_spksp_0']

    elif model_set_number in [4]:
        models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                       'prespos_0psh_1spksm_1_spksp_0',

                       'prespos_1psh_1spksm_0_spksp_0', 
                       'prespos_1psh_1spksm_1_spksp_0',
                       
                       'hist_1pos_1psh_1spksm_0_spksp_0', 
                       'hist_1pos_1psh_1spksm_1_spksp_0', 
                       
                       'hist_2pos_1psh_1spksm_0_spksp_0', 
                       'hist_2pos_1psh_1spksm_1_spksp_0']

    elif model_set_number in [7]:
        ### generalization
        if use_action:
            models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']
        else:
            models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']

    elif model_set_number in [8]:
        models_to_include = [
        'hist_1pos_0psh_0spksm_1_spksp_0', 
        #'hist_1pos_1psh_0spksm_0_spksp_0',
        'hist_1pos_3psh_1spksm_0_spksp_0']

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    if model_set_number in [1, 3, 4, 7, 8]:
        key = 'spks'

    elif model_set_number in [2]:
        key = 'np'

    if load_file is None:
        ### get the model predictions: 
        model_individual_cell_tuning_curves(animal=animal, history_bins_max=4, 
            ridge=True, include_action_lags = True, return_models = True, 
            models_to_include = models_to_include)
    
    elif load_file == 'default':
        ### load the models: 
        if use_action:
            model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, True), 'rb'))
        else:
            if model_set_number == 7:
                model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, False), 'rb'))
            model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %(model_set_number), 'rb'))

        if model_set_number == 7:
            print 'separating task-specific models'
            modz = []; 
            for mod in models_to_include:
                assert model_dict[0, mod].shape[2] == 3
                for i_t, tsk in enumerate(['co', 'obs','gen']):
                    for i_d in np.arange(ndays[animal]):
                        model_dict[i_d, mod+tsk] = model_dict[i_d, mod][:, :, i_t]
                    modz.append(mod+tsk)
            models_to_include = modz; 

        if model_set_number == 8:
            ### Goal is to compare y_t | y_{t-1} vs. y_t | E(y_{t-1}|s_{t-2})
            mod_state_pos = 'hist_1pos_3psh_0spksm_0_spksp_0'
            # mod_state     = 'hist_1pos_1psh_0spksm_0_spksp_0'
            mod_dyn = 'hist_1pos_0psh_0spksm_1_spksp_0'


            mod0 = 'STATE_all_targ_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            #mod1 = 'STATE_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            # mod2 = 'RESIDUAL_POS_ONLY_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            # mod3 = 'RESIDUAL_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            mod_dyn2 = models_to_include[0]
            mod_id = 'Identity'
            #models_to_include = [mod_dyn2, mod_dyn2+'potent', mod_dyn2+'null',  mod_id, mod_id+'potent', mod_id+'null']
            models_to_include = [mod_dyn2, mod0]#, mod1, mod_id]#, mod2, mod3]

            for i_d in np.arange(ndays[animal]):

            #     ### Setup zeros
                #model_dict[i_d, mod_dyn2] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod0] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                # model_dict[i_d, mod1] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod_id] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                #model_dict[i_d, mod_id, 'pot'] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                #model_dict[i_d, mod_id, 'null'] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                
                # model_dict[i_d, mod2] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                # model_dict[i_d, mod3] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                
                bin_num = model_dict[i_d, 'bin_num']
                spks = model_dict[i_d, 'spks']
                N = len(bin_num)

                # ix = np.nonzero(bin_num > np.min(bin_num))[0]
                # model_dict[i_d, mod_id][ix, :] = spks[ix - 1, :]
                
                # if animal == 'grom':
                #     KG, KG_null, KG_pot = get_KG_decoder_grom(i_d)

                # if animal == 'jeev':
                #     KG, KG_null, KG_pot = get_KG_decoder_jeev(i_d)
            
                # model_dict[i_d, mod_id, 'null'][ix, :] = np.dot(KG_null, spks[ix-1, :].T).T
                # model_dict[i_d, mod_id, 'pot'][ix, :] = np.dot(KG_pot, spks[ix-1, :].T).T

                ### Assert that only skipping the bin indices #1
                #ix_nan = np.isnan(model_dict[i_d, mod_id][:, 0])
                #assert np.all(bin_num[ix_nan] == 1)

                ### Go through the folds: 
                for i_fold in range(5):

                    ### Get the testing indices: 
                    fold_ix = model_dict[i_d, mod_dyn, i_fold, 'model_testix']
                    fold_ix = fold_ix[fold_ix < N - n_steps_prop]

                    ix_keep = np.nonzero(bin_num[fold_ix + n_steps_prop] >= n_steps_prop + 1)[0] ## Want bin nums >= 2 bc if 1, then didn't have 0 to prop. 
                    fold_ix = fold_ix[ix_keep]

                    ### Get the dynamics model: 
                    dyn_model_ = model_dict[i_d, mod_dyn, i_fold, 'model']
                    #state_model_intc = model_dict[i_d, mod_state, i_fold, 'model'].intercept_
                    state_model_pos_intc = model_dict[i_d, mod_state_pos, i_fold, 'model'].intercept_

                    ### Get the tru data: 
                    y_true = model_dict[i_d, 'spks'][fold_ix, :]
                    y_dyn2 = y_true.copy()

                    ### Get the estimated y_t | s_{t-1}; 
                    #y_state = model_dict[i_d, mod_state][fold_ix, :]
                    y_state_pos = model_dict[i_d, mod_state_pos][fold_ix, :]

                    ### Y_residual: 
                    #y_res = y_true - y_state + state_model_intc[np.newaxis, :]; 
                    y_res_pos = y_true - y_state_pos + state_model_pos_intc[np.newaxis, :];  

                    ### Using the state & propogating. 
                    ## Propogate: 
                    for n in range(n_steps_prop):
                        y_dyn2 = dyn_model_.predict(y_dyn2)
                        y_state_pos = dyn_model_.predict(y_state_pos)
                        #y_state = dyn_model_.predict(y_state)
                        #y_res_pos = dyn_model_.predict(y_res_pos)
                        #y_res = dyn_model_.predict(y_res)

                    model_dict[i_d, mod_dyn2][fold_ix + n_steps_prop] = y_dyn2; 
                    model_dict[i_d, mod0][fold_ix+n_steps_prop, :] = y_state_pos; 
                    # model_dict[i_d, mod1][fold_ix+n_steps_prop, :] = y_state; 
                    # model_dict[i_d, mod2][fold_ix+n_steps_prop, :] = y_res_pos; 
                    # model_dict[i_d, mod3][fold_ix+n_steps_prop, :] = y_res; 
            
            
    hist_bins = np.arange(15)

    ### Pooled over all days: 
    f_all, ax_all = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (15, 6))
    
    TD_all = dict();
    PD_all = dict(); 
    for mod in models_to_include:
        for sig in ['all', 'sig']:
            TD_all[mod, sig] = []; 
            PD_all[mod, sig] = []; 
        
    for i_d in range(ndays[animal]):
        
        ### Only include days where the position is within a specific radius; 
        if animal == 'grom':
            rad = np.sqrt(np.sum(model_dict[i_d, 'pos']**2, axis=1))
            keep_ix = np.nonzero(rad <= restrict_radius)[0]
            model_dict[i_d, 'keep_ix_pos'] = keep_ix
        
        elif animal == 'jeev':
            if restrict_radius > 100:
                model_dict[i_d, 'keep_ix_pos'] = np.arange(model_dict[i_d, 'pos'].shape[0])
            else:
                ftest, axtest = plt.subplots(ncols = 3, nrows = 3)
                obs = np.nonzero(model_dict[i_d, 'task'] == 1)[0]
                trg = model_dict[i_d, 'trg'][obs]
                trls = model_dict[i_d, 'trl'][obs]

                for i, it in enumerate(np.unique(trg)):
                    ix_tg = np.nonzero(trg == it)[0]

                    ## Get individual trials: 
                    trls_tg = np.unique(trls[ix_tg])
                    for ii, itrl_trg in enumerate(trls_tg):

                        ### Get indices for individual trial for target: 
                        ix_trl_tg = np.nonzero(trls == itrl_trg)[0]

                        pos = model_dict[i_d, 'pos'][obs[ix_trl_tg], :]
                        axi = axtest[i/3, i%3]
                        axi.plot(pos[:, 0], pos[:, 1], '-', color=cmap_list[int(i)], linewidth=2.0)

                    # binz = np.min(model_dict[i_d, 'bin_num'][obs[ix_tg]]) + 2
                    # binz_ix = np.nonzero(model_dict[i_d, 'bin_num'][obs[ix_tg]] == binz)[0]
                    # axtest.plot(model_dict[i_d, 'pos'][obs[ix_tg[binz_ix]], 0], 
                    #     model_dict[i_d, 'pos'][obs[ix_tg[binz_ix]], 1], '*', color=cmap_list[int(it)])
                
        ### Basics -- get the binning for the neural push commands: 
        neural_push = model_dict[i_d, 'np'][model_dict[i_d, 'keep_ix_pos'], :]

        ### Commands
        commands = subspace_overlap.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
        
        ### Get task / target
        tsk = model_dict[i_d, 'task'][model_dict[i_d, 'keep_ix_pos']]
        targ = model_dict[i_d, 'trg'][model_dict[i_d, 'keep_ix_pos']]
        bin_num = model_dict[i_d, 'bin_num'][model_dict[i_d, 'keep_ix_pos']]

        ### Now go through each task targ and assess when there are enough observations: 
        try:
            y_true = 10*model_dict[i_d, key][model_dict[i_d, 'keep_ix_pos'], :]
        except:
            if key == 'spks':
                y_true = 10*model_dict[i_d, 'spikes'][model_dict[i_d, 'keep_ix_pos'], :]
            else:
                raise Exception

        T, N = y_true.shape

        if important_neurons:
            important_neur = imp_file[(i_d, animal, 'svd')]
        else:
            important_neur = np.arange(N); 

        if skip_ind_plots:
            important_neur = []; 

        f_dict = dict(); 

        ### Get the decoder 
        ### Get the decoder ###
        if animal == 'grom':
            KG, KG_null, KG_pot = get_KG_decoder_grom(i_d)

        if animal == 'jeev':
            KG, KG_null, KG_pot = get_KG_decoder_jeev(i_d)

        # if np.logical_or(only_null, only_potent):
        #     if np.logical_and(only_null, only_potent):
        #         raise Exception('Cant have null and potent')
        #     else:
        #         y_true = get_decomp_y(KG_null, KG_pot, y_true, only_null = only_null, only_potent=only_potent)
        
        ### Make the spiking histograms: 
        ### [xvel, yvel] x [mag] x [ang] x [task1-targ] x [task2-targ]

        sig_diff = np.zeros((N, 4, 8, 8, 8)) -1

        index_dictionary = {}; 

        for i_mag in range(4):
            for i_ang in range(8):
                for i_tsk1_tg in range(8):

                    if within_task:
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == within_task_task) & (targ == i_tsk1_tg)
                        next_targ_set = np.arange(i_tsk1_tg+1, 8)
                    else:
                        ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0) & (targ == i_tsk1_tg)
                        next_targ_set = np.arange(8)

                    if next_pt_pred:
                        ix0 = np.nonzero(ix0 == True)[0]
                        ix0 = ix0 + n_steps_prop; 
                        ix0 = ix0[ix0 < len(tsk)]
                        ix0 = ix0[bin_num[ix0] > 0]

                    else:
                        ## Find relevant commands: 
                        ix0 = np.nonzero(ix0 == True)[0]

                    for i_tsk2_tg in next_targ_set:
                        if within_task:
                            ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == within_task_task) & (targ == i_tsk2_tg)
                        else:
                            ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1) & (targ == i_tsk2_tg)
                        
                        if next_pt_pred:
                            ix1 = np.nonzero(ix1 == True)[0]
                            ix1 = ix1 + n_steps_prop; 
                            ix1 = ix1[ix1 < len(tsk)]
                            ix1 = ix1[bin_num[ix1] > 0]

                        else:
                            ## Find relevant commands: 
                            ix1 = np.nonzero(ix1 == True)[0]
                        
                        if np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs):

                            index_dictionary[i_mag, i_ang, i_tsk1_tg, i_tsk2_tg] = [ix0, ix1]
                            
                            for n in range(N):
                                _, pv = scipy.stats.ks_2samp(y_true[ix0, n], y_true[ix1, n])
                                sig_diff[n, i_mag, i_ang, i_tsk1_tg, i_tsk2_tg] = pv < 0.05; 


        ###################################
        ### Now get the diff models: ######
        print 'Done with day -- minus 1 --'
        sig = ['k', 'r']

        f, ax = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (9, 6))
        for axi in ax.reshape(-1):
            axi.axis('square')

        ### Rows are CO / OBS; 
        f_, ax_ = plt.subplots(ncols = len(models_to_include), nrows = 4, figsize = (3*len(models_to_include), 8))
        for i_m, mod in enumerate(models_to_include):

            #### These are using the null/potent 
            if 'potent' in mod:
                modi = mod[:-6]
                y_pred = 10*model_dict[i_d, modi, 'pot']
            
            elif 'null' in mod:
                modi = mod[:-4]
                y_pred = 10*model_dict[i_d, modi, 'null']
            
            else:
                y_pred = 10*model_dict[i_d, mod][model_dict[i_d, 'keep_ix_pos'], :]

            # ### Get null activity ###
            # if np.logical_or(only_null, only_potent):
            #     y_pred = get_decomp_y(KG_null, KG_pot, y_pred, only_null = only_null, only_potent=only_potent)

            ax[0, i_m].set_title(mod, fontsize=6)
            TD = []; PD = []; TD_s = []; PD_s = [];

            pred = []; obs = []; 
            predkg = []; obskg = [];
            diff_tru = []; diff_trukg = []; 
            diff_pred = []; diff_predkg = []; 

            ### Make a plot
            for i_mag in range(4):
                for i_ang in range(8):
                    for i_tsk1_tg in range(8):
                        for i_tsk2_tg in range(8):
    
                            if tuple([i_mag, i_ang, i_tsk1_tg, i_tsk2_tg]) in index_dictionary.keys():
                                #print 'Plotting %d, %d, %d, %d' %(i_mag, i_ang, i_tsk1_tg, i_tsk2_tg)
                                ix0, ix1 = index_dictionary[i_mag, i_ang, i_tsk1_tg, i_tsk2_tg]

                                assert np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs)
                        
                                if key == 'spks':
                                    #### True difference over all neurons -- CO vs. OBS:
                                    tru_co = np.nanmean(y_true[ix0, :], axis=0)
                                    tru_ob = np.nanmean(y_true[ix1, :], axis=0)

                                    pred_co = np.nanmean(y_pred[ix0, :], axis=0)
                                    pred_ob = np.nanmean(y_pred[ix1, :], axis=0)

                                    if cov:
                                        tru_diff = get_cov_diffs(ix0, ix1, y_true, [], method = 1, mult=.1)
                                        pred_diff = get_cov_diffs(ix0, ix1, y_pred, [], method = 1, mult=.1)
                                        # tru_diff = [tru_diff[0]]
                                        # pred_diff = [pred_diff[0]]
                                        # import pdb; pdb.set_trace()
                                    else:
                                        tru_diff = tru_co - tru_ob
                                        pred_diff = pred_co - pred_ob

                                elif key == 'np':
                                    ### Mean angle: 
                                    # mean_co = math.atan2(np.mean(y_true[ix0, 1]), np.mean(y_true[ix0, 0]))
                                    # mean_ob = math.atan2(np.mean(y_true[ix1, 1]), np.mean(y_true[ix1, 0]))
                                    
                                    # pred_mean_co = math.atan2(sw.mean(y_pred[ix0, 1]), np.mean(y_pred[ix0, 0]))
                                    # pred_mean_ob = math.atan2(np.mean(y_pred[ix1, 1]), np.mean(y_pred[ix1, 0]))
                                    
                                    # ### do an angular difference: 
                                    # tru_diff = ang_difference(np.array([mean_co]), np.array([mean_ob]))
                                    # pred_diff = ang_difference(np.array([pred_mean_co]), np.array([pred_mean_ob]))
                                    tru_co = np.nanmean(y_true[ix0, :], axis=0)
                                    tru_ob = np.nanmean(y_true[ix1, :], axis=0)

                                    pred_co = np.nanmean(y_pred[ix0, :], axis=0)
                                    pred_ob = np.nanmean(y_pred[ix1, :], axis=0)

                                    tru_diff = tru_co - tru_ob
                                    pred_diff = pred_co - pred_ob

                                for n, (td, pd) in enumerate(zip(tru_diff, pred_diff)):
                                    sig_bin = int(sig_diff[n, i_mag, i_ang, i_tsk1_tg, i_tsk2_tg])
                                    
                                    if sig_bin == 1:
                                        # ax[1, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=1.)
                                        # ax[0, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=1.)

                                        # ax_all[1, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=1.)
                                        # ax_all[0, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=1.)

                                        TD_s.append(td);
                                        PD_s.append(pd);

                                        TD_all[mod, 'sig'].append(td);
                                        PD_all[mod, 'sig'].append(pd);

                                    # ax[0, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=1.)
                                    # ax_all[0, i_m].plot(td, pd, '.', color = sig[sig_bin], alpha=0.2)

                                TD.append(tru_diff);
                                PD.append(pred_diff);
                                TD_all[mod, 'all'].append(tru_diff);
                                PD_all[mod, 'all'].append(pred_diff);
                                    
                                if plot_pred_vals:

                                    diff_tru.append(tru_co - tru_ob)
                                    diff_pred.append(pred_co - pred_ob)

                                    pred.append(pred_co)
                                    pred.append(pred_ob)
                                    
                                    obs.append(tru_co)
                                    obs.append(tru_ob)

                                    if key != 'np':

                                        if kg_diff_angle:
                                            #### TRUE  ####
                                            ang_co = np.squeeze(np.array(np.dot(KG, tru_co.T)))
                                            ang_co = math.atan2(ang_co[1], ang_co[0])

                                            ang_ob = np.squeeze(np.array(np.dot(KG, tru_ob.T)))
                                            ang_ob = math.atan2(ang_ob[1], ang_ob[0])
                                            diff_trukg.append(ang_difference(np.array([ang_co]), np.array([ang_ob]))[:, np.newaxis])

                                            #### PREDICTED ####
                                            ang_co = np.squeeze(np.array(np.dot(KG, pred_co.T)))
                                            ang_co = math.atan2(ang_co[1], ang_co[0])

                                            ang_ob = np.squeeze(np.array(np.dot(KG, pred_ob.T)))
                                            ang_ob = math.atan2(ang_ob[1], ang_ob[0])
                                            diff_predkg.append(ang_difference(np.array([ang_co]), np.array([ang_ob]))[:, np.newaxis])

                                            predkg.append(np.dot(KG, pred_co.T))
                                            predkg.append(np.dot(KG, pred_ob.T))

                                            obskg.append(np.dot(KG, tru_co.T))
                                            obskg.append(np.dot(KG, tru_ob.T))         

                                        else:
                                            diff_trukg.append(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T))
                                            diff_predkg.append(np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T))

                                            predkg.append(np.dot(KG, pred_co.T))
                                            predkg.append(np.dot(KG, pred_ob.T))

                                            obskg.append(np.dot(KG, tru_co.T))
                                            obskg.append(np.dot(KG, tru_ob.T))

                                            # cokg = np.nanmean(neural_push[ix0, :], axis=0)
                                            # obkg = np.nanmean(neural_push[ix1, :], axis=0)

                                            # assert np.sum(np.abs(cokg - np.dot(KG, tru_co.T))) < 5e5
                                            # assert np.sum(np.abs(obkg - np.dot(KG, tru_ob.T))) < 5e5

                                    ax_[0, i_m].set_title('True Y_t vs. Pred Y_t\nModel %s' %mod)
                                    ax_[1, i_m].set_title('True K*Y_t vs. Pred K*Y_t\nModel %s' %mod)
                                #import pdb; pdb.set_trace()

            if plot_pred_vals:

                ### Predicted 
                P = [np.hstack((pred)).reshape(-1), np.hstack((diff_pred)).reshape(-1)]

                ### Observed
                O = [np.hstack((obs)).reshape(-1), np.hstack((diff_tru)).reshape(-1)]
                
                ### Get population R2: 
                Ppop = [np.vstack((pred)), np.vstack((diff_pred))]

                ### Observed
                Opop = [np.vstack((obs)), np.vstack((diff_tru))]

                R2s = []
                for ii in range(2):
                    R2s.append(get_R2(Opop[ii], Ppop[ii], pop = True, ignore_nans = True))

                ### Indices
                ix = [0, 2, ]

                for ii, (p, o, iixx, vaf) in enumerate(zip(P, O, ix, R2s)):
                    ix = np.nonzero(np.logical_and(~np.isnan(p), ~np.isnan(o)))[0]
                    ax_[iixx, i_m].plot(np.squeeze(np.array(o[ix])), np.squeeze(np.array(p[ix])), '.', markersize = 2.)
                    
                    ### Get slopes ###
                    slp,intc,rv,pv,err = scipy.stats.linregress(np.array(o[ix]), np.squeeze(np.array(p[ix])))
                    x_ = np.linspace(np.min(o[ix]), np.max(o[ix]), 100)
                    y_ = slp*x_ + intc; 
                    ax_[iixx, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[iixx, i_m].text(np.percentile(x_, 70), np.percentile(y_, 50),
                        'r = %.2f \npv = %.2f\nslp = %.2f\nvaf pop=%.3f' %(rv, pv, slp, vaf))
                

                if key != 'np':
                    P2 = [np.squeeze(np.array(np.hstack((predkg)).reshape(-1))), 
                        np.squeeze(np.array(np.hstack((diff_predkg)))).reshape(-1)]

                    O2 = [np.squeeze(np.array(np.hstack((obskg)).reshape(-1))), 
                        np.squeeze(np.array(np.hstack((diff_trukg)))).reshape(-1)]

                    P2pop = [np.squeeze(np.array(np.vstack((predkg)))), 
                             np.squeeze(np.array(np.vstack((diff_predkg))))]

                    O2pop = [np.squeeze(np.array(np.vstack((obskg)))), 
                             np.squeeze(np.array(np.vstack((diff_trukg))))]

                    R2s_np = []
                    for ii in range(2):
                        R2s_np.append(get_R2(O2pop[ii], P2pop[ii], pop = True, ignore_nans = True))

                    ix = [1, 3]

                    ylim = [[-2, 2], [-.5, .5]]

                    for ii, (p, o, iixx, yl, vaf) in enumerate(zip(P2, O2, ix, ylim, R2s_np)):
                        ixa = np.nonzero(np.logical_and(~np.isnan(p), ~np.isnan(o)))[0]
                        ax_[iixx, i_m].plot(np.squeeze(np.array(o[ixa])), np.squeeze(np.array(p[ixa])), '.', markersize = 2.)
                        
                        ### Get slopes ###
                        slp,intc,rv,pv,err = scipy.stats.linregress(np.array(o[ixa]), np.squeeze(np.array(p[ixa])))

                        x_ = np.linspace(np.min(o[ix]), np.max(o[ix]), 100)
                        y_ = slp*x_ + intc; 
                        ax_[iixx, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                        ax_[iixx, i_m].text(np.percentile(x_, 70), np.percentile(y_, 50),
                            'r = %.2f \npv = %.2f\nslp = %.2f\nVAF pop = %.2f' %(rv, pv, slp, vaf), color='r')
                        #ax_[iixx, i_m].set_ylim(yl)

                # if key == 'spks':
                #     ax_[2, i_m].set_ylim([-2, 2])
                # elif key == 'np':
                #     ax_[2, i_m].set_ylim([-.52, .52])

                assert(np.allclose(np.hstack((TD)), np.hstack((diff_tru))))
                assert(np.allclose(np.hstack((PD)), np.hstack((diff_pred))))
            
            ### Lets do a linear correlation: 
            
            x = np.vstack((TD)); 
            y = np.vstack((PD))
            vaf = get_R2(x, y, pop = True, ignore_nans = False)
            
            #vaf = VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)
            print('Day : %d, Model %s, pop VAF: %.3f' %(i_d, mod, vaf))

            slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD)), np.hstack((PD)))
            x_ = np.linspace(np.min(np.hstack((TD))), np.max(np.hstack((TD))), 100)
            y_ = slp*x_ + intc; 
            ax[0, i_m].plot(x_, y_, '-', color='gray', linewidth = 1.5)
            ax[0, i_m].plot(np.hstack((TD)), np.hstack((PD)), 'k.', alpha=1., markersize=1., color='gray')
            try:
                if cov: 
                    pass
                else:
                    ax[0, i_m].plot(np.hstack((TD_s)), np.hstack((PD_s)), 'k.', markersize = 1)
            except:
                pass
            ax[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f\nvaf pop=%.3f' %(rv, pv, slp, vaf),
                fontsize=9, color='b')

            if cov: 
                ax[0, i_m].set_xlim([-250, 250])
                ax[0, i_m].set_ylim([-250, 250])
                ax[0, i_m].plot([-250, 250], [-250, 250], 'k--', linewidth = 1.)
                
            else:
                ax[0, i_m].set_xlim([-40, 40])
                ax[0, i_m].set_ylim([-40, 40])
                ax[0, i_m].plot([-40, 40], [-40, 40], 'k--', linewidth = 1.)

            if cov:
                pass
            else:
                slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD_s)), np.hstack((PD_s)))
                x_ = np.linspace(np.min(np.hstack((TD_s))), np.max(np.hstack((TD_s))), 100)
                y_ = slp*x_ + intc; 
                ax[1, i_m].plot(np.hstack((TD_s)), np.hstack((PD_s)), 'r.')
                #ax[0, i_m].plot(np.hstack((TD_s)), np.hstack((PD_s)), 'r.')
                ax[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                ax[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f' %(rv, pv, slp))
                
            #ax[0, i_m].set_ylim([-2, 3])
            if cov: 
                pass
            else:
                pass#ax[1, i_m].set_ylim([-2, 3])
            print('Done with day -- %d --'%i_m)
        
        f.tight_layout()
        try:
            f_.tight_layout()
            #f.savefig(savedir+animal+'_day_%d_trudiff_vs_preddiff_xtask_xcond_mean_corrs_model_set%d_xcond_min_obs%.2f_rad_lte_%.2f.svg'%(i_d, model_set_number, min_obs, restrict_radius))
        except:
            pass
        # if within_task:
        #     f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_within_task%d.png' %within_task_task)
        # else:
        #     f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_xcond.png')

        print 'Done with day -- end --'

    ### Get all the stuff for the ax_all; 
    for i_m, mod in enumerate(models_to_include):
        for sig, (sig_nm, alph, alph_col) in enumerate(zip(['all', 'sig'], [.3, 1.0], ['k', 'r'])):
            x = np.hstack((TD_all[mod, sig_nm]))
            y = np.hstack((PD_all[mod, sig_nm]))

            ### get variance explained: 
            VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)
            #VAF_ind = 1 - np.sum((x-y)**2, axis=0) / np.sum((x-np.mean(x, axis=0)[np.newaxis, :])**2, axis=0)

            ax_all[sig, i_m].plot(x, y, '.', color=alph_col, alpha=alph, markersize=2)
            if sig == 1:
                ax_all[sig-1, i_m].plot(x, y, '.', color=alph_col, alpha=alph, markersize=2)

            slp,intc,rv,pv,err = scipy.stats.linregress(x, y)
            x_ = np.linspace(np.min(x), np.max(x), 100)
            y_ = slp*x_ + intc; 
            ax_all[sig, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            ax_all[sig, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f\nvaf gross =%.2f' %(rv, pv, slp, VAF), color='b')
            if cov: 
                pass
            else:
                pass#ax_all[sig, i_m].set_ylim([-2, 2.])

    f_all.tight_layout()
    # if within_task:
    #     f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_within_task%d.png' %within_task_task)
    # else:
    #     f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_xcond.png')

def mean_diffs_plot_x_condition_state_limited(animal = 'grom', min_obs = 15,
    model_set_number = 8, ndays = None, limit_to_center = False, restrict_radius = 1000., 
    divide_by_vel = True, ndiv = 16.):

    load_file = 'default'; 
    important_neurons = True; 
    only_null = False; 
    only_potent = False; 
    skip_ind_plots = True; 
    next_pt_pred = True
    plot_pred_vals = True;
    n_steps_prop = 1;
    use_action = False; 
    within_task = False; 

    '''
    same as mean_diffs_plot, now taking mean differences across condition 
        - only do across conditions that are across task; 
    search for commands that are the SAME. 
        - predict their NEXT time step. 
        - what is the mean diff in their NEXT time step vs. predicted diff in NEXT time step
    '''
    
    savedir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

    ### Magnitude boundaries: 
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    marks = ['-', '--']

    if important_neurons:
        if animal == 'grom':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/grom_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))
        
        elif animal == 'jeev':
            imp_file = pickle.load(open('/Users/preeyakhanna/fa_analysis/online_analysis/jeev_important_neurons_svd_feb2019_thresh_0.8.pkl', 'rb'))

    ### List the models to analyze

    if model_set_number in [8]:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0', 
        'hist_1pos_3psh_0spksm_0_spksp_0']

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=np.arange(9), jeev=[0, 2, 3])
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    if model_set_number in [1, 3, 4, 7, 8]:
        key = 'spks'

    elif model_set_number in [2]:
        key = 'np'

    if load_file is None:
        ### get the model predictions: 
        model_individual_cell_tuning_curves(animal=animal, history_bins_max=4, 
            ridge=True, include_action_lags = True, return_models = True, 
            models_to_include = models_to_include)
    
    elif load_file == 'default':
        ### load the models: 
        if use_action:
            model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, True), 'rb'))
        else:
            if model_set_number == 7:
                model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, False), 'rb'))
            model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %(model_set_number), 'rb'))

        if model_set_number == 7:
            print 'separating task-specific models'
            modz = []; 
            for mod in models_to_include:
                assert model_dict[0, mod].shape[2] == 3
                for i_t, tsk in enumerate(['co', 'obs','gen']):
                    for i_d in np.arange(ndays[animal]):
                        model_dict[i_d, mod+tsk] = model_dict[i_d, mod][:, :, i_t]
                    modz.append(mod+tsk)
            models_to_include = modz; 

        if model_set_number == 8:
            ### Goal is to compare y_t | y_{t-1} vs. y_t | E(y_{t-1}|s_{t-2})
            mod_state_pos = 'hist_1pos_3psh_0spksm_0_spksp_0'
            mod_state     = 'hist_1pos_1psh_0spksm_0_spksp_0'
            mod_dyn = 'hist_1pos_0psh_0spksm_1_spksp_0'

            mod0 = 'STATE_POS_ONLY_targ_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            mod1 = 'STATE_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            mod2 = 'RESIDUAL_POS_ONLY_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            mod3 = 'RESIDUAL_rad_%.2f_minobs_%.1f'%(restrict_radius, min_obs)
            mod_dyn2 = 'spks_nsteps_%d' %(n_steps_prop)
            models_to_include = [mod_dyn2, mod1]#, mod0, mod2, mod3]

            for i_d in ndays[animal]:

                ### Setup zeros
                model_dict[i_d, mod_dyn2] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod0] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod1] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod2] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                model_dict[i_d, mod3] = np.zeros_like(model_dict[i_d, mod_dyn]) + np.nan
                
                bin_num = model_dict[i_d, 'bin_num']
                N = len(bin_num)

                ### Go through the folds: 
                for i_fold in range(5):

                    ### Get the testing indices: 
                    fold_ix = model_dict[i_d, mod_dyn, i_fold, 'model_testix']
                    fold_ix = fold_ix[fold_ix < N - n_steps_prop]

                    ix_keep = np.nonzero(bin_num[fold_ix + n_steps_prop] > n_steps_prop + 1)[0] ## Want bin nums >= 2 bc if 1, then didn't have 0 to prop. 
                    fold_ix = fold_ix[ix_keep]

                    ### Get the dynamics model: 
                    dyn_model_ = model_dict[i_d, mod_dyn, i_fold, 'model']
                    state_model_intc = model_dict[i_d, mod_state, i_fold, 'model'].intercept_
                    state_model_pos_intc = model_dict[i_d, mod_state_pos, i_fold, 'model'].intercept_

                    ### Get the tru data: 
                    y_true = model_dict[i_d, 'spks'][fold_ix, :]
                    y_dyn2 = y_true.copy()

                    ### Get the estimated y_t | s_{t-1}; 
                    y_state = model_dict[i_d, mod_state][fold_ix, :]
                    y_state_pos = model_dict[i_d, mod_state_pos][fold_ix, :]

                    ### Y_residual: 
                    y_res = y_true - y_state + state_model_intc[np.newaxis, :]; 
                    y_res_pos = y_true - y_state_pos + state_model_pos_intc[np.newaxis, :];  

                    ### Using the state & propogating. 
                    ## Propogate: 
                    for n in range(n_steps_prop):
                        y_dyn2 = dyn_model_.predict(y_dyn2)
                        y_state_pos = dyn_model_.predict(y_state_pos)
                        y_state = dyn_model_.predict(y_state)
                        y_res_pos = dyn_model_.predict(y_res_pos)
                        y_res = dyn_model_.predict(y_res)

                    model_dict[i_d, mod_dyn2][fold_ix + n_steps_prop] = y_dyn2; 
                    model_dict[i_d, mod0][fold_ix+n_steps_prop, :] = y_state_pos; 
                    model_dict[i_d, mod1][fold_ix+n_steps_prop, :] = y_state; 
                    model_dict[i_d, mod2][fold_ix+n_steps_prop, :] = y_res_pos; 
                    model_dict[i_d, mod3][fold_ix+n_steps_prop, :] = y_res; 

    ### Pooled over all days: 
    f_all, ax_all = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (15, 6))
    
    TD_all = dict();
    PD_all = dict(); 
    
    for mod in models_to_include:
        for sig in ['all', 'sig']:
            TD_all[mod, sig] = []; 
            PD_all[mod, sig] = []; 
        
    for i_d in ndays[animal]:
        
        if limit_to_center:
            ### Only include days where the position is within a specific radius; 
            if animal == 'grom':
                rad = np.sqrt(np.sum(model_dict[i_d, 'pos']**2, axis=1))
                keep_ix = np.nonzero(rad <= restrict_radius)[0]
                model_dict[i_d, 'keep_ix_pos', 0] = keep_ix
                ncats = 1; 
            elif animal == 'jeev': 
                raise Exception 

        elif divide_by_vel: 
            
            #### Get velocities at t=-1 ###
            vel_tm1 = model_dict[i_d, 'vel_tm1']; 

            ### Get angles out from this velocity: 
            vel_disc = subspace_overlap.commands2bins([vel_tm1], mag_boundaries, animal, i_d, vel_ix = [0, 1], ndiv = ndiv)[0]

            ### Here, the second column (vel_disc[:, 1]) has info on the angle: 
            for ang_i in range(int(ndiv)): 
                ix = np.nonzero(vel_disc[:, 1] == ang_i)[0]
                model_dict[i_d, 'keep_ix_pos', ang_i] = ix; 
            ncats = int(ndiv);

        ### Get the decoder 
        ### Get the decoder ###
        if animal == 'grom':
            KG, KG_null, KG_pot = get_KG_decoder_grom(i_d)

        if animal == 'jeev':
            KG, KG_null, KG_pot = get_KG_decoder_jeev(i_d)

        sig_diff = np.zeros((ncats, model_dict[i_d, 'spks'].shape[1], 4, 8, 10, 10)) -1

        DIFFs = dict(); 
        PREDs = dict(); 

        f, ax = plt.subplots(ncols = len(models_to_include), nrows = 2, figsize = (15, 6))
        f_, ax_ = plt.subplots(ncols = len(models_to_include), nrows = 4, figsize = (3*len(models_to_include), 8))
        
        for model in models_to_include:
            DIFFs[model, 'td'] = []
            DIFFs[model, 'pd'] = []
            DIFFs[model, 'td_s'] = []
            DIFFs[model, 'pd_s'] = []

            PREDs[model, 'pred'] = []; 
            PREDs[model, 'true'] = []; 

            PREDs[model, 'pred_kg'] = []; 
            PREDs[model, 'true_kg'] = []; 

            PREDs[model, 'diff_pred_kg'] = []; 
            PREDs[model, 'diff_true_kg'] = []; 

            PREDs[model, 'diff_pred'] = []; 
            PREDs[model, 'diff_true'] = [];                 
        

        ### Now go through each vel category: 
        for VEL_STATE in range(ncats):

            ix_keeping = model_dict[i_d, 'keep_ix_pos', VEL_STATE]

            if len(ix_keeping) > 0:

                ### Basics -- get the binning for the neural push commands: 
                neural_push = model_dict[i_d, 'np'][ix_keeping, :]

                ### Commands
                commands = subspace_overlap.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
            
                ### Get task / target
                tsk = model_dict[i_d, 'task'][ix_keeping]
                targ = model_dict[i_d, 'trg'][ix_keeping]
                bin_num = model_dict[i_d, 'bin_num'][ix_keeping]

                ### Now go through each task targ and assess when there are enough observations: 
                try:
                    y_true = model_dict[i_d, key][ix_keeping, :]
                except:
                    if key == 'spks':
                        y_true = model_dict[i_d, 'spikes'][ix_keeping, :]
                    else:
                        raise Exception

                T, N = y_true.shape

                if important_neurons:
                    important_neur = imp_file[(i_d, animal, 'svd')]
                else:
                    important_neur = np.arange(N); 

                if skip_ind_plots:
                    important_neur = []; 

                index_dictionary = {}; 

                for i_mag in range(4):
                    for i_ang in range(8):
                        for i_tsk1_tg in range(10):

                            if within_task:
                                ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == within_task_task) & (targ == i_tsk1_tg)
                                next_targ_set = np.arange(i_tsk1_tg+1, 8)
                            else:
                                ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 0) & (targ == i_tsk1_tg)
                                next_targ_set = np.arange(10)

                            if next_pt_pred:
                                ix0 = np.nonzero(ix0 == True)[0]
                                ix0 = ix0 + n_steps_prop; 
                                ix0 = ix0[ix0 < len(tsk)]
                                ix0 = ix0[bin_num[ix0] > n_steps_prop ] # No zeros I think bc hist-1, so look for bin > n_steps
                            else:
                                ## Find relevant commands: 
                                ix0 = np.nonzero(ix0 == True)[0]

                            for i_tsk2_tg in next_targ_set:
                                if within_task:
                                    ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == within_task_task) & (targ == i_tsk2_tg)
                                else:
                                    ix1 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == 1) & (targ == i_tsk2_tg)
                                
                                if next_pt_pred:
                                    ix1 = np.nonzero(ix1 == True)[0]
                                    ix1 = ix1 + n_steps_prop; 
                                    ix1 = ix1[ix1 < len(tsk)]
                                    ix1 = ix1[bin_num[ix1] > 0]

                                else:
                                    ## Find relevant commands: 
                                    ix1 = np.nonzero(ix1 == True)[0]
                                
                                if np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs):

                                    index_dictionary[i_mag, i_ang, i_tsk1_tg, i_tsk2_tg] = [ix0, ix1]
                                    
                                    for n in range(N):
                                        _, pv = scipy.stats.ks_2samp(y_true[ix0, n], y_true[ix1, n])
                                        sig_diff[VEL_STATE, n, i_mag, i_ang, i_tsk1_tg, i_tsk2_tg] = pv < 0.05; 


                ###################################
                ### Now get the diff models: ######
                print 'Done with day -- minus 1 --'
                sig = ['k', 'r']

                ### Go through each model
                for i_m, mod in enumerate(models_to_include):

                    #### These are using the null/potent 
                    if 'potent' in mod:
                        modi = mod[:-6]
                        y_pred = model_dict[i_d, modi, 'pot']
                    
                    elif 'null' in mod:
                        modi = mod[:-4]
                        y_pred = model_dict[i_d, modi, 'null']
                    
                    else:
                        y_pred = model_dict[i_d, mod][model_dict[i_d, 'keep_ix_pos', VEL_STATE], :]

                    
                    ax[0, i_m].set_title(mod, fontsize=6)

                    ### Make a plot
                    for i_mag in range(4):
                        for i_ang in range(8):
                            for i_tsk1_tg in range(8):
                                for i_tsk2_tg in range(8):
            
                                    if tuple([i_mag, i_ang, i_tsk1_tg, i_tsk2_tg]) in index_dictionary.keys():
                                        #print 'Plotting %d, %d, %d, %d' %(i_mag, i_ang, i_tsk1_tg, i_tsk2_tg)
                                        ix0, ix1 = index_dictionary[i_mag, i_ang, i_tsk1_tg, i_tsk2_tg]

                                        assert np.logical_and(len(ix0) >= min_obs, len(ix1) >= min_obs)
                                
                                        if key == 'spks':
                                            #### True difference over all neurons -- CO vs. OBS:
                                            tru_co = np.nanmean(y_true[ix0, :], axis=0)
                                            tru_ob = np.nanmean(y_true[ix1, :], axis=0)

                                            pred_co = np.nanmean(y_pred[ix0, :], axis=0)
                                            pred_ob = np.nanmean(y_pred[ix1, :], axis=0)

                                            tru_diff = tru_co - tru_ob
                                            pred_diff = pred_co - pred_ob

                                        elif key == 'np':
                                            tru_co = np.nanmean(y_true[ix0, :], axis=0)
                                            tru_ob = np.nanmean(y_true[ix1, :], axis=0)

                                            pred_co = np.nanmean(y_pred[ix0, :], axis=0)
                                            pred_ob = np.nanmean(y_pred[ix1, :], axis=0)

                                            tru_diff = tru_co - tru_ob
                                            pred_diff = pred_co - pred_ob

                                        for n, (td, pd) in enumerate(zip(tru_diff, pred_diff)):
                                            sig_bin = int(sig_diff[VEL_STATE, n, i_mag, i_ang, i_tsk1_tg, i_tsk2_tg])
                                            
                                            if sig_bin == 1:

                                                DIFFs[mod, 'td_s'].append(td)
                                                DIFFs[mod, 'pd_s'].append(pd)

                                            DIFFs[mod, 'td_s'].append(td)
                                            DIFFs[mod, 'pd_s'].append(pd)
                                            

                                        if plot_pred_vals:

                                            PREDs[mod, 'pred'].append(pred_co)
                                            PREDs[mod, 'pred'].append(pred_ob)
                                            
                                            PREDs[mod, 'true'].append(tru_co)
                                            PREDs[mod, 'true'].append(tru_ob) 

                                            PREDs[mod, 'diff_pred'].append(pred_co - pred_ob)
                                            PREDs[mod, 'diff_true'].append(tru_co - tru_ob)

                                            if key != 'np':
                                                PREDs[mod, 'pred_kg'].append(np.dot(KG, pred_co.T))
                                                PREDs[mod, 'pred_kg'].append(np.dot(KG, pred_ob.T))

                                                PREDs[mod, 'true_kg'].append(np.dot(KG, tru_co.T))
                                                PREDs[mod, 'true_kg'].append(np.dot(KG, tru_ob.T))

                                                PREDs[mod, 'diff_pred_kg'].append(np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T))
                                                PREDs[mod, 'diff_true_kg'].append(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T))

                                                #diff_trukg.append(np.dot(KG, tru_co.T) - np.dot(KG, tru_ob.T))
                                                #diff_predkg.append(np.dot(KG, pred_co.T)-np.dot(KG, pred_ob.T))

                                                cokg = np.nanmean(neural_push[ix0, :], axis=0)
                                                obkg = np.nanmean(neural_push[ix1, :], axis=0)
                                                assert np.sum(np.abs(cokg - np.dot(KG, tru_co.T))) < 5e5
                                                assert np.sum(np.abs(obkg - np.dot(KG, tru_ob.T))) < 5e5


        if plot_pred_vals:
            
            ### Go through each model
            for i_m, mod in enumerate(models_to_include):

                ### Predicted 
                P = [np.vstack((PREDs[mod, 'pred'])), np.vstack((PREDs[mod, 'diff_pred']))]
                O = [np.vstack((PREDs[mod, 'true'])), np.vstack((PREDs[mod, 'diff_true']))]

                vaf = []; 
                for q in range(2):
                    vaf.append(get_R2(O[q], P[q], pop = True, ignore_nans = True))

                ### Indices
                ix = [0, 2]

                for ii, (p, o, iixx, vafi) in enumerate(zip(P, O, ix, vaf)):
                    p = p.reshape(-1)
                    o = o.reshape(-1)
                    ix = np.nonzero(np.logical_and(~np.isnan(p), ~np.isnan(o)))[0]
                    ax_[iixx, i_m].plot(np.squeeze(np.array(o[ix])), np.squeeze(np.array(p[ix])), '.', markersize = 2.)
                    
                    ### Get slopes ###
                    slp,intc,rv,pv,err = scipy.stats.linregress(np.array(o[ix]), np.squeeze(np.array(p[ix])))

                    x_ = np.linspace(np.min(o[ix]), np.max(o[ix]), 100)
                    y_ = slp*x_ + intc; 
                    ax_[iixx, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                    ax_[iixx, i_m].text(np.percentile(x_, 70), np.percentile(y_, 50),
                        'r = %.2f \npv = %.2f\nslp = %.2f\nvaf pop=%.2f' %(rv, pv, slp, vafi), color='b')
                
                if key != 'np':
                    P2 = [np.squeeze(np.array(np.vstack((PREDs[mod, 'pred_kg'])))), 
                        np.squeeze(np.array(np.vstack((PREDs[mod, 'diff_pred_kg']))))]

                    O2 = [np.squeeze(np.array(np.vstack((PREDs[mod, 'true_kg'])))), 
                        np.squeeze(np.array(np.vstack((PREDs[mod, 'diff_true_kg']))))]
                    ix = [1, 3]
                    
                    vaf2 = []; 
                    for q in range(2):
                        vaf2.append(get_R2(O2[q], P2[q], pop = True, ignore_nans = True))

                    for ii, (p, o, iixx, vaf2i) in enumerate(zip(P2, O2, ix, vaf2)):
                        p = p.reshape(-1)
                        o = o.reshape(-1)
                        ix = np.nonzero(np.logical_and(~np.isnan(p), ~np.isnan(o)))[0]
                        ax_[iixx, i_m].plot(np.squeeze(np.array(o[ix])), np.squeeze(np.array(p[ix])), '.', markersize = 2.)
                        
                        ### Get slopes ###
                        slp,intc,rv,pv,err = scipy.stats.linregress(np.array(o[ix]), np.squeeze(np.array(p[ix])))

                        x_ = np.linspace(np.min(o[ix]), np.max(o[ix]), 100)
                        y_ = slp*x_ + intc; 
                        ax_[iixx, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
                        ax_[iixx, i_m].text(np.percentile(x_, 70), np.percentile(y_, 50),
                            'r = %.2f \npv = %.2f\nslp = %.2f\nVAF pop = %.2f' %(rv, pv, slp, vaf2i), color='r')

                if key == 'spks':
                    ax_[2, i_m].set_ylim([-2, 2])
                elif key == 'np':
                    ax_[2, i_m].set_ylim([-.52, .52])

            # ### Lets do a linear correlation: 
            # slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD)), np.hstack((PD)))
            # x_ = np.linspace(np.min(np.hstack((TD))), np.max(np.hstack((TD))), 100)
            # y_ = slp*x_ + intc; 
            # ax[0, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            # ax[0, i_m].plot(np.hstack((TD)), np.hstack((PD)), 'k.', alpha=.2)
            # ax[0, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f' %(rv, pv, slp))

            # slp,intc,rv,pv,err = scipy.stats.linregress(np.hstack((TD_s)), np.hstack((PD_s)))
            # x_ = np.linspace(np.min(np.hstack((TD_s))), np.max(np.hstack((TD_s))), 100)
            # y_ = slp*x_ + intc; 
            # ax[1, i_m].plot(np.hstack((TD_s)), np.hstack((PD_s)), 'r.')
            # ax[0, i_m].plot(np.hstack((TD_s)), np.hstack((PD_s)), 'r.')
            # ax[1, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
            # ax[1, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f' %(rv, pv, slp))
            
            # ax[0, i_m].set_ylim([-2, 3])
            # ax[1, i_m].set_ylim([-2, 3])
            # print('Done with day -- %d --'%i_m)
            
        #f.tight_layout()
        try:
            f_.tight_layout()
            #f.savefig(savedir+animal+'_day_%d_trudiff_vs_preddiff_xtask_xcond_mean_corrs_model_set%d_xcond_min_obs%.2f_rad_lte_%.2f.png'%(i_d, model_set_number, min_obs, restrict_radius))
        except:
            pass
        # if within_task:
        #     f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_within_task%d.png' %within_task_task)
        # else:
        #     f.savefig(savedir+animal+'_day_'+str(i_d) + '_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_xcond.png')

        print 'Done with day -- end --'

    # ### Get all the stuff for the ax_all; 
    # for i_m, mod in enumerate(models_to_include):
    #     for sig, (sig_nm, alph, alph_col) in enumerate(zip(['all', 'sig'], [.3, 1.0], ['k', 'r'])):
    #         x = np.hstack((TD_all[mod, sig_nm]))
    #         y = np.hstack((PD_all[mod, sig_nm]))

    #         ### get variance explained: 
    #         VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)
    #         #VAF_ind = 1 - np.sum((x-y)**2, axis=0) / np.sum((x-np.mean(x, axis=0)[np.newaxis, :])**2, axis=0)

    #         ax_all[sig, i_m].plot(x, y, '.', color=alph_col, alpha=alph, markersize=2)
    #         if sig == 1:
    #             ax_all[sig-1, i_m].plot(x, y, '.', color=alph_col, alpha=alph, markersize=2)

    #         slp,intc,rv,pv,err = scipy.stats.linregress(x, y)
    #         x_ = np.linspace(np.min(x), np.max(x), 100)
    #         y_ = slp*x_ + intc; 
    #         ax_all[sig, i_m].plot(x_, y_, '-', color='gray', linewidth = .5)
    #         ax_all[sig, i_m].text(np.percentile(x_, 70), np.percentile(y_, 20), 'r = %.2f \npv = %.2f\nslp = %.2f\nvaf=%.2f' %(rv, pv, slp, VAF), color='b')
    #         ax_all[sig, i_m].set_ylim([-2, 2.])

    # f_all.tight_layout()
    # if within_task:
    #     f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_within_task%d.png' %within_task_task)
    # else:
    #     f_all.savefig(savedir+animal+'_all_days_trudiff_vs_preddiff_xtask_xcond_mean_corrs_onlyNull_'+str(only_null)+'onlyPotent'+str(only_potent)+'_model_set'+str(model_set_number)+'_xcond.png')

### Plot variance and cov difference plots? 
def fit_ridge(y_train, data_temp_dict, x_var_names, test_data=None, test_data2=None, train_data2 = None,
    alpha = 1.0, only_potent_predictor = False, KG_pot = None, fit_task_specific_model_test_task_spec = False):

    model_2 = Ridge(alpha=alpha)
    x = []
    for vr in x_var_names:
        x.append(data_temp_dict[vr][: , np.newaxis])
    X = np.hstack((x))

    X2 = []
    if train_data2 is not None: 
        for vr in x_var_names:
            X2.append(train_data2[vr][:, np.newaxis])
        X2 = np.hstack((X2))
        X = np.vstack((X, X2))

    if only_potent_predictor:
        X = np.dot(KG_pot, X.T).T
        print 'only potetn used to train ridge'

    if fit_task_specific_model_test_task_spec:
        tsk = data_temp_dict['tsk']
        ix0 = np.nonzero(tsk == 0)[0]
        ix1 = np.nonzero(tsk == 1)[0]

        X0 = X[ix0, :]; X1 = X[ix1, :]; 
        Y0 = y[ix0, :]; Y1 = y[ix1, :];
        model_2.fit(X0, Y0); 
        model_2_ = Ridge(alpha=alpha)
        model_2_.fit(X1, Y1); 
        model_2 = [model_2, model_2_]

    else:
        model_2.fit(X, y_train)
        model_2.X = X
        model_2.y = y_train
        model_2.coef_names = x_var_names

    if test_data is None:
        return model_2
    else:
        y = []
        z = []
        for vr in x_var_names:
            y.append(test_data[vr][:, np.newaxis])
            if test_data2 is not None:
                z.append(test_data2[vr][:, np.newaxis])
        Y = np.hstack((y))
        if test_data2 is not None:
            Z = np.hstack((z))
        
        pred = model_2.predict(Y)
        if test_data2 is not None:
            pred2 = model_2.predict(Z)
        else:
            pred2 = None
        
        return model_2, pred, pred2

def fit_ridge_2step(data_all, data_spec, data_test, variables, alpha=0., alpha_state = 0., 
    alpha_common = 0., predict_key = 'spks'):

    ######################################################################
    ############ Fit the model on specific target state info first #######
    ######################################################################
    print 'target info for Grom only!'
    target_dats = sio.loadmat('/Users/preeyakhanna/fa_analysis/online_analysis/unique_targ.mat')
    target_dats = target_dats['unique_targ']

    spks_all = data_all[predict_key]
    spks_spec = data_spec[predict_key]
    spks_spec_test = data_test[predict_key]

    ### Going to fit everything except state; 
    model_gen = Ridge(alpha=alpha)
    
    x_all = []; ### This has all non-state info. 
    x_all_state = []; 

    x_spec_nonstate = []; ### This has all non-state info for targ-specific
    x_spec_state = [];  ### This has all state info for targ-specific. 

    x_common_train = []; ### This is basically x_all, but includes state info, used for training common model later/ 
    x_common_test = []; 

    x_test_nonstate = []; 
    x_test_state = []; 

    N1 = data_all['bin_num'].shape[0]
    N2 = data_spec['bin_num'].shape[0]
    N3 = data_test['bin_num'].shape[0]

    for vr in variables:

        ### Just take the state data (should be less than the number of pts below)
        if vr == 'trg':
            print('Adding target')
            ### Get the target info; 
            targ_pos = target_dats[data_all['trg'].astype(int), :]
            x_all_state.append(targ_pos)

            targ_pos1 = target_dats[data_spec['trg'].astype(int), :]
            x_spec_state.append(targ_pos1)

            targ_pos2 = target_dats[data_test['trg'].astype(int), :]
            x_test_state.append(targ_pos2)

        elif np.logical_or('pos' in vr, 'vel' in vr):

            ### Data specific to task/target; 
            x_all_state.append(data_all[vr][:, np.newaxis])
            x_spec_state.append(data_spec[vr][:, np.newaxis])
            x_test_state.append(data_test[vr][:, np.newaxis])
            
            x_common_train.append(data_all[vr][:, np.newaxis])
            x_common_test.append(data_test[vr][:, np.newaxis])

        #### All data, should be more data than before
        else:
            x_all.append(data_all[vr][:, np.newaxis])
            x_spec_nonstate.append(data_spec[vr][:, np.newaxis])
            x_test_nonstate.append(data_test[vr][:, np.newaxis])
            x_common_train.append(data_all[vr][:, np.newaxis])
            x_common_test.append(data_test[vr][:, np.newaxis])

    
    ### State train / test
    X_all_state = np.hstack((x_all_state))
    X_spec_state = np.hstack((x_spec_state))
    X_test_state = np.hstack((x_test_state))
    
    X_common_train = np.hstack((x_common_train))
    X_common_test = np.hstack((x_common_test))

    try:
        X = np.hstack((x_all))
        X_spec_nonstate = np.hstack((x_spec_nonstate))
        X_test_nonstate = np.hstack((x_test_nonstate))
        only_gen_vs_spec = False
    except:
        only_gen_vs_spec = True

    ###     If there is other info besides task info
    ### fit that generally and try fitting state info as task-spec or not
    ###     Else, just fit it on other stuff. 

    if only_gen_vs_spec:
        ### General model: 
        model_gen.fit(X_all_state, spks_all)
        pred_Y2 = model_gen.predict(X_test_state)

        ### Task specific model: 
        model_spec = Ridge(alpha=alpha)
        model_spec.fit(X_spec_state, spks_spec)
        pred_Y = model_spec.predict(X_test_state)
        print('State specific')
    else:

        model_gen.fit(X, spks_all)

        ### Now use the residuals of this model and fit them with specifci state model; 
        spks_pred = model_gen.predict(X_spec_nonstate)
        spks_resid = spks_spec - spks_pred

        ### Fit the residuals with a model using state: 
        model_spec = Ridge(alpha=alpha_state)
        model_spec.fit(X_spec_state, spks_resid)

        ### Try out test data: 
        ######################

        ### Predict the spikes w/ A
        spks_pred_test = model_gen.predict(X_test_nonstate)

        ### Compute the residuals
        spks_pred_resid = spks_spec_test - spks_pred_test

        ### Predict the residuals with B
        Y_res = model_spec.predict(X_test_state)

        ### Estimate the neural state: add the estimation 
        pred_Y = spks_pred_test + Y_res

        ### Get R2 for different conditons ###
        r2_spks1 = 1 - (np.sum((spks_pred_test - spks_spec_test)**2, axis=0)/np.sum((spks_spec_test - np.mean(spks_spec_test, axis=0)[np.newaxis, :])**2))
        r2_spks2 = 1 - (np.sum((Y_res - spks_spec_test)**2, axis=0)/np.sum((spks_spec_test - np.mean(spks_spec_test, axis=0)[np.newaxis, :])**2))
        r2_spks3 = 1 - (np.sum((pred_Y - spks_spec_test)**2, axis=0)/np.sum((spks_spec_test - np.mean(spks_spec_test, axis=0)[np.newaxis, :])**2))

        ################################################################
        ############ Fit the model on target state info jointly  #######
        ################################################################
        
        ### Also do a 2-step here; 
        model_com2 = Ridge(alpha=alpha)
        model_com2.fit(X, spks_all)

        spks_pred = model_com2.predict(X)
        spks_resid = spks_all - spks_pred; 

        model_spec2 = Ridge(alpha=alpha_state)
        model_spec2.fit(X_all_state, spks_resid)

        ### make a predition for the same test data as above: 
        pred_Y_neur = model_com2.predict(X_test_nonstate)
        pred_Y_comm = model_spec2.predict(X_test_state)
        pred_Y2 = pred_Y_neur + pred_Y_comm
    #r2_spks0 = 1 - (np.sum((pred_Y_comm - spks_spec_test)**2, axis=0)/np.sum((spks_spec_test - np.mean(spks_spec_test, axis=0)[np.newaxis, :])**2))

    ### Now get the two models: 
    return model_gen, model_spec, pred_Y, pred_Y2

### Plot prediction R2s for model 2 -- a_t | a_{t-1}, y_{t}
def plot_r2_bar_model_2():
    f, ax = plt.subplots(ncols = 2, nrows = 2)

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):

        hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/%s%s/%s_models_to_pred_mn_diffs_model_set2.h5' %(animal, yr, animal))
        models_to_include = [
                            'prespos_0psh_0spksm_1_spksp_0', # current spikes
                            'hist_1pos_0psh_0spksm_1_spksp_0', # past spikes
                            'hist_1pos_0psh_1spksm_0_spksp_0', ] # past push
                            # 'hist_1pos_0psh_1spksm_1_spksp_0', # past push, past y_t
                            # 'hist_4pos_0psh_1spksm_0_spksp_0',  # past push x 4
                            # 'hist_4pos_0psh_1spksm_1_spksp_0', # past push x 4, past y_t
                            # 'hist_4pos_0psh_1spksm_4_spksp_0']; # past push x 4, past y_t x 4

        xlab = [
        'y_{t}',
        'y_{t-1}',
        'a_{t-1}',
        'a_{t-1}, y_{t-1}',
        'a_{t-4}...a_{t-1}', 
        'a_{t-4}...a_{t-1}, y_{t-1}', 
        'a_{t-4}...a_{t-1}, y_{t-4}...y_{t-1}']
    
        fold = ['maroon', 'orangered', 'goldenrod', 'teal', 'blue']

        for d, xy in enumerate(['x', 'y']):

            for i_m, mod in enumerate(models_to_include):
                r2 = []; 

                for i_f in range(5):
                    tbl = getattr(hdf.root, mod+'_fold_'+str(i_f))
                    
                    for i_d in np.unique(tbl[:]['day_ix']):
                        ix = np.nonzero(tbl[:]['day_ix'] == i_d)[0]
                        r2.append(tbl[ix[d]]['r2'])
                        ax[d, ia].plot(i_m + np.random.randn()*.05, float(tbl[ix[d]]['r2']), '.', color=fold[int(i_d)], alpha=.8)

                r2 = np.hstack((r2)).reshape(-1)
                ax[d, ia].bar(i_m, np.mean(r2))
                ax[d, ia].errorbar(i_m, np.mean(r2), np.std(r2)/np.sqrt(len(r2)), marker='|', color='k')
            ax[d, ia].set_ylabel('Predicted Push (a_t)')

            ax[d, ia].set_xticks(np.arange(len(xlab)))
            ax[d, ia].set_xticklabels(xlab,rotation=90)
            ax[d, ia].set_title('Monk %s'%animal)

### Figure 4 -- behavior bar plots; 
def fig_4_behavoir_tuning():
    for r2_pop in [True, False]:
        for perc_increase in [True, False]:
            plot_r2_bar_model_1(r2_pop = r2_pop, perc_increase = perc_increase)

### Fig 5 -- identity vs. neural dynamcis
def fig_5_neural_dyn(min_obs = 15, r2_pop = True, perc_increase = False, 
    model_set_number = 3, ndays = None,):
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    fsumm, axsumm = plt.subplots(ncols = 2, nrows = 1, figsize = (4, 4))

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 3: 
            models_to_include = [#'prespos_0psh_0spksm_1_spksp_0',
                                 #'prespos_0psh_0spksm_1_spksp_0potent',
                                 #'prespos_0psh_0spksm_1_spksp_0null',
                                 'hist_1pos_0psh_0spksm_1_spksp_0',
                                 'identity']
                                 #'hist_1pos_0psh_1spksm_0_spksp_0']
                                 #'hist_1pos_0psh_0spksm_1_spksp_0potent',
                                 #'hist_1pos_0psh_0spksm_1_spksp_0null']
            models_colors = [[39, 169, 225], [0, 0, 0]]
            xlab = [['$y_{t-1} | y_{t}$'], ['$a_{t-1} | y_{t}$']]

        M = len(models_to_include)
        models_colors = [np.array(m)/256. for m in models_colors]
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        R2S = dict(); R2_push = dict()
        ### Super basic R2 plot of neural activity and neural push ###

        for i_d in range(ndays[animal]):

            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

            ###### True data #####
            tdata = model_dict[i_d, 'spks']
            np_data = model_dict[i_d, 'np']
            bin_num = model_dict[i_d, 'bin_num']
            min_bin = np.min(bin_num)

            R2s = dict()

            ###### Get the baseline ####
            for i_mod, mod in enumerate(models_to_include):
                if mod == 'identity':
                    ix = np.nonzero(bin_num>min_bin)[0]
                    pdata = np.zeros_like(tdata) + np.nan 
                    pdata[ix, :] = tdata[ix - 1]

                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np_data[ix - 1, :]

                    ### Get population R2 ### 
                    R2 = get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)
                
                else:
                    pdata = model_dict[i_d, mod]
                    np_pred = np.squeeze(np.array(np.dot(KG, pdata.T).T))

                    ### Get population R2 ### 
                    R2 = get_R2(tdata, pdata, pop = r2_pop)

                    ### Get push ####
                    R2_np = get_R2(np_data, np_pred, pop = r2_pop)

                ### Put the R2 in a dictionary: 
                R2S[i_mod, i_d, 0] = R2; 
                R2S[i_mod, i_d, 1] = R2_np; 

            ##### Plot this single day #####
            for q in range(2): 
                tmp = []; 
                for i_mod in range(M):
                    try:
                        tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d, q]))))
                    except:
                        tmp.append(R2S[i_mod, i_d, q])
                axsumm[q].plot(ia*3 + np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        for q in range(2):
            tmp = []; tmp_e = []; 
            
            for i_mod in range(M):
                tmp2 = []; 
                for i_d in range(ndays[animal]):
                    tmp2.append(R2S[i_mod, i_d, q])
                tmp2 = np.hstack((tmp2))
                tmp2 = tmp2[~np.isnan(tmp2)]

                ## mean 
                tmp.append(np.mean(tmp2))

                ## s.e.m
                tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

            ### Overal mean 
            for i_mod in range(M):
                axsumm[q].bar(ia*3 + i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
                axsumm[q].errorbar(ia*3 + i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
            axsumm[q].set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase), fontsize=8)

            axsumm[q].set_xticks(np.arange(M))
            axsumm[q].set_xticklabels(xlab[q], rotation=45, fontsize=6)

    fsumm.tight_layout()
    fsumm.savefig(fig_dir + 'both_%sr2_dyn_model_perc_increase%s_model%d.svg'%(pop_str, perc_increase, model_set_number))

### Fig 5 -- ID vs. neural dynamics on next action diffs. 
def fig_5_neural_dyn_mean_pred(min_obs = 15, r2_pop = True, perc_increase = False, 
    model_set_number = 3, ndays = None, state_limit = True, ndiv = 16, center_limit = False,
    center_rad_limit = None, jeev_days = [0, 2, 3]):
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=np.arange(9), jeev=jeev_days)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    fsumm, axsumm = plt.subplots(ncols = 3, nrows = 1, figsize = (8, 4))

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 3: 
            models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0',
                                 'identity']

        elif model_set_number == 8:
            models_to_include = ['identity',
                                 'hist_1pos_1psh_0spksm_0_spksp_0',
                                 'hist_1pos_3psh_0spksm_0_spksp_0',
                                 'hist_1pos_0psh_0spksm_1_spksp_0']

            models_colors = [[0, 0, 0], [101, 44, 144], [101, 44, 144], [39, 169, 225]]
            xlab = [['$y_{t+1} | y_{t}$'], ['$a_{t+1} | y_{t}$'], ['$\Delta a_{t+1} | y_{t}$']]

        M = len(models_to_include)
        models_colors = [np.array(m)/256. for m in models_colors]
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        R2S = dict(); R2_push = dict()
        ### Super basic R2 plot of neural activity and neural push ###

        for i_d in ndays[animal]:

            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

            ###### True data #####
            tdata = model_dict[i_d, 'spks']
            np_data = model_dict[i_d, 'np']
            target = model_dict[i_d, 'trg']
            task = model_dict[i_d, 'task']

            ### Get commands: 
            commands_disc = subspace_overlap.commands2bins([np_data], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            bin_num = model_dict[i_d, 'bin_num']
            min_bin = np.min(bin_num)

            R2s = dict()

            ###### Get the baseline ####
            for i_mod, mod in enumerate(models_to_include):
                
                if mod == 'identity':
                    ix = np.nonzero(bin_num>min_bin)[0]
                    pdata = np.zeros_like(tdata) + np.nan 
                    pdata[ix, :] = tdata[ix - 1, :]

                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np_data[ix - 1, :]

                    ### Get population R2 ### 
                    R2 = get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)

                elif mod in ['hist_1pos_1psh_0spksm_0_spksp_0', 'hist_1pos_3psh_0spksm_0_spksp_0']:
                    ix = np.nonzero(bin_num>min_bin)[0]
                    pdata = np.zeros_like(tdata) + np.nan 

                    ### Get states
                    state_preds = model_dict[i_d, mod][ix - 1, :]

                    ### Propagate by the dynamics model: choose first fold: 
                    model_dyn = model_dict[i_d, 'hist_1pos_0psh_0spksm_1_spksp_0', 0, 'model']
                    pdata[ix, :] = model_dyn.predict(state_preds)

                    ixnan = np.nonzero(np.isnan(pdata[:, 0]))
                    assert(np.all(bin_num[ixnan] == 1))

                    np_pred = np.zeros_like(np_data) + np.nan
                    np_pred[ix, :] = np.dot(KG, pdata[ix, :].T).T

                    ### Get population R2 ### 
                    R2 = get_R2(tdata, pdata, pop = r2_pop, ignore_nans = True)

                    ### Get push ####
                    R2_np = get_R2(np_data, np_pred, pop = r2_pop, ignore_nans = True)

                else:
                    pdata = model_dict[i_d, mod]
                    np_pred = np.squeeze(np.array(np.dot(KG, pdata.T).T))

                    ### Get population R2 ### 
                    R2 = get_R2(tdata, pdata, pop = r2_pop)

                    ### Get push ####
                    R2_np = get_R2(np_data, np_pred, pop = r2_pop)
                
                ### Put the R2 in a dictionary: 
                R2S[i_mod, i_d, 0] = R2; 
                R2S[i_mod, i_d, 1] = R2_np; 
                
                ### Find the mean DIFFS 
                ### Go through the targets (1)
                ### Got through targets (2)
                true_diffs_kg = []; 
                pred_diffs_kg = []; 

                KEEP_IXs = {}

                if state_limit:
                    if center_limit:
                        rad = np.sqrt(np.sum(model_dict[i_d, 'pos']**2, axis=1))
                        keep_ix = np.nonzero(rad <= center_rad_limit)[0]
                        KEEP_IXs[i_d, 'keep_ix_pos', 0] = keep_ix
                        ncats = 1; 
                    
                    else:
                        #### Get velocities at t=-1 ###
                        vel_tm1 = model_dict[i_d, 'vel_tm1']; 

                        ### Get angles out from this velocity: 
                        vel_disc = subspace_overlap.commands2bins([vel_tm1], mag_boundaries, animal, i_d, vel_ix = [0, 1], ndiv = ndiv)[0]

                        ### Here, the second column (vel_disc[:, 1]) has info on the angle: 
                        for ang_i in range(int(ndiv)): 
                            ix = np.nonzero(vel_disc[:, 1] == ang_i)[0]
                            KEEP_IXs[i_d, 'keep_ix_pos', ang_i] = ix; 
                        ncats = int(ndiv);

                for cat in range(ncats):
                    ix_keep = KEEP_IXs[i_d, 'keep_ix_pos', cat]
                    
                    sub_commands_disc = commands_disc[ix_keep, :]
                    sub_target = target[ix_keep]
                    sub_task = task[ix_keep]

                    for i_ang in range(8):
                        for i_mag in range(4):
                            for itg in range(10):

                                ix_co = (sub_commands_disc[:, 0] == i_mag) & (sub_commands_disc[:, 1] == i_ang) & (sub_target == itg) & (sub_task == 0)
                                ix_co = np.nonzero(ix_co == True)[0]

                                ix_co_big = ix_keep[ix_co]
                                ix_co_big = ix_co_big + 1;

                                ix_co_big = ix_co_big[ix_co_big < len(bin_num)]
                                ix_co_big = ix_co_big[bin_num[ix_co_big] > 1]
                                
                                if len(ix_co_big) >= min_obs: 
        
                                    for itg2 in range(10):
                                        ix_ob = (sub_commands_disc[:, 0] == i_mag) & (sub_commands_disc[:, 1] == i_ang) & (sub_target == itg2) & (sub_task == 1)
                                        ix_ob = np.nonzero(ix_ob == True)[0]
                                        ix_ob_big = ix_keep[ix_ob]

                                        ix_ob_big = ix_ob_big + 1; 
                                        ix_ob_big = ix_ob_big[ix_ob_big < len(bin_num)]
                                        ix_ob_big = ix_ob_big[bin_num[ix_ob_big] > 1]  

                                        if len(ix_ob_big) >= min_obs: 

                                            true_diffs_kg.append( np.nanmean(np_data[ix_co_big, :], axis=0) - np.nanmean(np_data[ix_ob_big, :], axis=0))
                                            pred_diffs_kg.append( np.nanmean(np_pred[ix_co_big, :], axis=0) - np.nanmean(np_pred[ix_ob_big, :], axis=0))

                ### Get R2: 
                R2S[i_mod, i_d, 2] = get_R2(np.vstack((true_diffs_kg)), np.vstack((pred_diffs_kg)), pop=r2_pop)


            ##### Plot this single day #####
            for q in range(3): 
                tmp = []; 
                for i_mod in range(M):
                    try:
                        tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d, q]))))
                    except:
                        tmp.append(R2S[i_mod, i_d, q])
                axsumm[q].plot(ia*5 + np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        for q in range(3):
            tmp = []; tmp_e = []; 
            
            dayz = []; 
            modz = []; 
            metz = []; 

            for i_mod in range(M):
                tmp2 = []; 
                for i_d in ndays[animal]:
                    tmp2.append(R2S[i_mod, i_d, q])

                    dayz.append(i_d)
                    modz.append(R2S[i_mod, i_d, q])
                    metz.append(i_mod)

                tmp2 = np.hstack((tmp2))
                tmp2 = tmp2[~np.isnan(tmp2)]

                ## mean 
                tmp.append(np.mean(tmp2))

                ## s.e.m
                tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

            dayz = np.hstack((dayz))
            modz = np.hstack((modz))
            metz = np.hstack((metz))

            pv, slp = run_LME(dayz, modz, metz)
            print('Animal: %s, Plot %d, PV: %.2f, SLP: %.2f' %(animal, q, pv, slp))

            ### Overal mean 
            for i_mod in range(M):
                axsumm[q].bar(ia*5 + i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
                axsumm[q].errorbar(ia*5 + i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
            axsumm[q].set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase), fontsize=8)

            axsumm[q].set_xticks(np.arange(M))
            axsumm[q].set_xticklabels(xlab[q], rotation=45, fontsize=6)

    fsumm.tight_layout()
    fsumm.savefig(fig_dir + 'both_%sfig_5_neural_dyn_mean_pred_model%d.svg'%(pop_str, model_set_number))
    
def plot_r2_bar_model_1(min_obs = 15, ndays = None, pt_2 = False, r2_pop = True, perc_increase = True, model_set_number = 1):
    
    ### For stats each neuron is an observation ##
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    key = 'spks' 

    ### Now generate plots -- testing w/ 1 day
    if ndays is None:
        ndays = dict(grom=9, jeev=4)
    else:
        ndays = dict(grom=ndays, jeev=ndays)

    for ia, (animal, yr) in enumerate(zip(['grom', 'jeev'], ['2016', '2013'])):
        
        if model_set_number == 1:
            models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                                'hist_1pos_-1psh_1spksm_0_spksp_0', 
                                'hist_1pos_1psh_1spksm_0_spksp_0',
                                'hist_1pos_2psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_0_spksp_0', 
                                'hist_1pos_3psh_1spksm_1_spksp_0']
            models_to_compare = np.array([0, 4, 5])

            models_colors = [[255., 0., 0.], 
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [101, 44, 144],
                             [39, 169, 225]]
            xlab = [
            '$a_{t}$',
            '$a_{t}, p_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk$',
            '$a_{t}, p_{t-1}, v_{t-1}, tg, tsk, y_{t-1}$']

        elif model_set_number == 3: 
            models_to_include = [#'prespos_0psh_0spksm_1_spksp_0',
                                 #'prespos_0psh_0spksm_1_spksp_0potent',
                                 #'prespos_0psh_0spksm_1_spksp_0null',
                                 'hist_1pos_0psh_0spksm_1_spksp_0',]
                                 #'hist_1pos_0psh_1spksm_0_spksp_0']
                                 #'hist_1pos_0psh_0spksm_1_spksp_0potent',
                                 #'hist_1pos_0psh_0spksm_1_spksp_0null']
            models_colors = [[39, 169, 225]]
            xlab = ['$y_{t-1}$']
            models_to_compare = []


        M = len(models_to_include)
        models_colors = [np.array(m)/256. for m in models_colors]

        fold = ['maroon', 'orangered', 'goldenrod', 'teal', 'blue']
        
        if r2_pop:
            pop_str = 'Population'
        else:
            pop_str = 'Indiv'        

        ### Go through each neuron and plot mean true vs. predicted (i.e R2) for a given command  / target combo: 
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl'%model_set_number, 'rb'))
            
        f, ax = plt.subplots(figsize=(4, 4))
        R2S = dict()

        ### Super basic R2 plot of neural activity and neural push ###
        R2s_ = dict()

        D = []; # Days; 
        Mod = []; # Model Number 
        R2_stats = []; ## Stats -- R2 for each neuron; 

        for i_d in range(ndays[animal]):

            if animal == 'grom': 
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)
            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

            ###### True data #####
            tdata = model_dict[i_d, 'spks']
            R2s = dict()

            ###### Get the baseline ####
            if perc_increase: 
                pdata = model_dict[i_d, models_to_include[0]]
                R2_baseline = get_R2(tdata, pdata, pop = r2_pop)

            for i_mod, mod in enumerate(models_to_include):
                pdata = model_dict[i_d, mod]

                ### Get population R2 ### 
                R2 = get_R2(tdata, pdata, pop = r2_pop)

                ### Only for indiv
                if i_mod in models_to_compare: 
                    if perc_increase:
                        if r2_pop:
                            pass
                        else:
                            assert(len(R2) == len(R2_baseline))
                        R2_stats.append((R2 - R2_baseline)/R2_baseline)
                    else:
                        R2_stats.append(R2)
                    # Remove NaNs 
                    D.append(np.zeros_like(R2) + i_d)
                    Mod.append(np.zeros_like(R2) + i_mod)

                ### Put the R2 in a dictionary: 
                if perc_increase:
                    R2S[i_mod, i_d] = (R2 - R2_baseline)/R2_baseline
                else:
                    R2S[i_mod, i_d] = R2; 

            ##### Plot this single day #####
            tmp = []; 
            for i_mod in range(M):
                try:
                    tmp.append(np.nanmean(np.hstack((R2S[i_mod, i_d]))))
                except:
                    tmp.append(R2S[i_mod, i_d])
            ax.plot(np.arange(M), tmp, '-', color='gray', linewidth = 1.)

        #### Plots total mean ###
        tmp = []; tmp_e = []; 
        for i_mod in range(M):
            tmp2 = []; 
            for i_d in range(ndays[animal]):
                tmp2.append(R2S[i_mod, i_d])
            tmp2 = np.hstack((tmp2))
            tmp2 = tmp2[~np.isnan(tmp2)]

            ## mean 
            tmp.append(np.mean(tmp2))

            ## s.e.m
            tmp_e.append(np.std(tmp2)/np.sqrt(len(tmp2)))

        ### Overal mean 
        for i_mod in range(M):
            ax.bar(i_mod, tmp[i_mod], color = models_colors[i_mod], edgecolor='k', linewidth = 1., )
            ax.errorbar(i_mod, tmp[i_mod], yerr=tmp_e[i_mod], marker='|', color='k')        
        ax.set_ylabel('%s R2, neur, perc_increase R2 %s'%(pop_str, perc_increase), fontsize=8)

        ax.set_xticks(np.arange(M))
        ax.set_xticklabels(xlab, rotation=45, fontsize=6)

        f.tight_layout()
        f.savefig(fig_dir + animal + '_%sr2_behav_models_perc_increase%s_model%d.svg'%(pop_str, perc_increase, model_set_number))
        
        ##### Print stats ####
        if model_set_number == 1:
            R2_stats = np.hstack((R2_stats))
            ix = ~np.isnan(R2_stats)

            ### Get non-nans: 
            R2_stats = R2_stats[ix]
            D = np.hstack((D))[ix]
            Mod = np.hstack((Mod))[ix]

            ### Run 2 LMEs: [0, 4] and [4, 5]; 
            for j, (m1, m2) in enumerate([[0, 4], [4, 5]]):
                ix = np.nonzero(np.logical_or(Mod == m1, Mod== m2))[0]

                pv, slp = run_LME(D[ix], Mod[ix], R2_stats[ix])
                print '---------------------------'
                print '---------------------------'
                print 'Pop R2 %s, Percent Increase %s'%(r2_pop, perc_increase)
                print 'LME: Animal %s, Model 1: %d, Model 2: %d, Pv: %.4f, Slp: %.4f' %(animal, m1, m2, pv, slp)
                print '---------------------------'
                print '---------------------------'
        if pt_2: 
            f1, ax1 = plt.subplots(ncols = 5, figsize=(12.5, 2.5))
            f2, ax2 = plt.subplots()
            ### now comput R2 ###
            Pred = dict(); Tru = dict(); 
            
            for i_d in range(ndays[animal]):

                ### Basics -- get the binning for the neural push commands: 
                neural_push = model_dict[i_d, 'np']

                ### Commands
                commands = subspace_overlap.commands2bins([neural_push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
                
                ### Get task / target
                tsk = model_dict[i_d, 'task']
                targ = model_dict[i_d, 'trg']
                bin_num = model_dict[i_d, 'bin_num']

                ### Now go through each task targ and assess when there are enough observations: 
                y_true = model_dict[i_d, key]
                T, N = y_true.shape

                ### Neural activity
                R2 = dict(); 
                for i_m, mod in enumerate(models_to_include):
                    R2['co', mod] = []; 
                    R2['obs', mod] = []; 
                    R2['both', mod] = []; 

                for i_mag in range(4):
                    for i_ang in range(8):
                        for i_t in range(2): # Task 
                            for targ in range(8): # target: 
                                ix0 = (commands[:,0] == i_mag) & (commands[:,1] == i_ang) & (tsk == i_t) & (targ == targ)
                                ix0 = np.nonzero(ix0 == True)[0]
                                if len(ix0) > min_obs:

                                    for i_m, model in enumerate(models_to_include):

                                        y_pred = model_dict[i_d, model]

                                        ### Get R2 of this observation: 
                                        spk_true = np.mean(y_true[ix0, :], axis=0)
                                        spk_pred = np.mean(y_pred[ix0, :], axis=0)

                                        if i_t == 0:
                                            R2['co', model].append([spk_true, spk_pred])
                                        elif i_t == 1:
                                            R2['obs', model].append([spk_true, spk_pred])

                                        ### Both 
                                        R2['both', model].append([spk_true, spk_pred])

                tsk_cols = ['b']#,'r','k']
                for i_t, tsk in enumerate(['both']):#, 'obs', 'both']):
                    for i_m, model in enumerate(models_to_include):

                        tru_co = np.vstack(( [R[0] for R in R2[tsk, model]] ))
                        pre_co = np.vstack(( [R[1] for R in R2[tsk, model]] ))

                        SSR = np.sum((tru_co - pre_co)**2)# Not over neruons, axis = 0)
                        SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2)#, axis=0)
                        R2_co_pop = 1 - (SSR/SST)
                        
                        SSR = np.sum((tru_co - pre_co)**2, axis = 0)
                        SST = np.sum((tru_co - np.mean(tru_co, axis=0)[np.newaxis, :])**2, axis=0)
                        R2_co_neur = 1 - (SSR/SST)

                        ax[0].plot(i_m, R2_co_pop, 'k*')
                        ax[0].set_ylabel('R2 -- of mean task/targ/command/neuron (population neuron R2)')
                        ax[0].set_ylim([-1, 1.])

                        ax[1].plot(i_m, np.nanmean(R2_co_neur), 'k*')
                        ax[1].set_ylabel('R2 -- of mean task/targ/command/neuron (individual neuron R2)')
                        ax[1].set_ylim([-1, 1.])

                        ax1[i_m].plot(tru_co.reshape(-1), pre_co.reshape(-1), 'k.', alpha=.2)
                        try:
                            Tru[model].append(tru_co.reshape(-1))
                            Pred[model].append(pre_co.reshape(-1))
                        except:
                            Tru[model] = [tru_co.reshape(-1)]
                            Pred[model] = [pre_co.reshape(-1)]
                            
            for i_m, model in enumerate(models_to_include):
                slp,intc,rv,pv,err =scipy.stats.linregress(np.hstack((Tru[model])), np.hstack((Pred[model])))
                x_ = np.linspace(np.min(np.hstack((Tru[model]))), np.max(np.hstack((Tru[model]))))
                y_ = slp*x_ + intc; 
                ax1[i_m].plot(x_, y_, '-', linewidth=.5)
                ax1[i_m].set_title('%s, \n pv=%.2f\nrv=%.2f\nslp=%.2f' %(model, pv, rv, slp),fontsize=8)


                if model == 'hist_1pos_0psh_0spksm_1_spksp_0':
                    y_pred = model_dict[i_d, model]
                    y_true = model_dict[i_d, key]; 

                    SSR = np.sum((y_pred - y_true)**2, axis=0)
                    SST = np.sum((y_true - np.mean(y_true, axis=0)[np.newaxis, :])**2, axis=0)
                    SST[np.isinf(SST)] = np.nan

                    r22 = 1 - SSR/SST; 
                    ax2.plot(r22)

                    print 'R2 mean day %d, %.2f', (i_d, np.mean(r22))

            #f.tight_layout()
            #f1.tight_layout()

def plot_r2_bar_model_5_gen(model_set_number = 5, ndays = None):
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    for animal in ['grom', 'jeev']:
        f, ax = plt.subplots(nrows = 2, ncols =2 )
        dat = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if ndays is None:
            if animal == 'grom':
                ndays = 9;
            elif animal == 'jeev':
                ndays = 4;

        for i_d in range(ndays):
            if model_set_number == 5:
                models_to_include = ['hist_1pos_1psh_0spksm_1_spksp_0', # past spikes
                                 'hist_1pos_1psh_1spksm_1_spksp_0', ]
                predict_key = 'spks'
            
            elif model_set_number == 6:
                models_to_include = ['hist_1pos_1psh_0spksm_0_spksp_0']
                predict_key = 'psh'

            task = dat[(i_d, 'task')]
            target = dat[(i_d, 'trg')]
            push = dat[(i_d, 'np')]
            spks = dat[(i_d, 'spikes')]
            mn_spks = np.mean(spks, axis=0)
            mn_push = np.mean(push, axis=0)

            if predict_key == 'spks':
                truth = spks; 
                truth_mn = mn_spks
            elif predict_key == 'psh':
                truth = push; 
                truth_mn = mn_push; 

            ### Get the KG: 
            if animal == 'grom':
                KG, _, _ = get_KG_decoder_grom(i_d)
            elif animal == 'jeev':
                KG, _, _ = get_KG_decoder_jeev(i_d)

            ### convert this to discrete commands
            commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            for i_m, mod in enumerate(models_to_include):

                ### for different tasks / targets plot R2s: 
                data_mod = dat[(i_d, mod)]
                data_mod_common = dat[(i_d, mod, 'common')]
                
                for di, (data, data_col) in enumerate(zip([data_mod, data_mod_common], ['b', 'k'])):
                    for tsk in range(2):
                        for targ in range(8):

                            ###### Estimated Neural activity ######
                            ix = (task == tsk) & (target == targ)
                            ix = np.nonzero(ix==True)[0]

                            SSR = np.sum((data[ix, :] - truth[ix, :])**2, axis=0); 
                            SST = np.sum((truth[ix, :] - truth_mn[np.newaxis, :])**2, axis=0)

                            R2 = 1 - (SSR/SST)
                            R2[np.isinf(R2)] = np.nan; 

                            ax[i_m, 0].bar(targ + 8*tsk + 0.4*di, np.nanmean(R2), color=data_col, width = .4)
                            ax[i_m, 0].set_title('R2 Neural Activity, \n %s' %mod)
                            
                            if predict_key == 'spks':
                                ###### Estimated Push ######
                                SSR = np.sum((np.dot(KG, data[ix, :].T).T - push[ix, :])**2, axis=0); 
                                SST = np.sum((push[ix, :] - mn_push[np.newaxis, :])**2, axis=0)
                                R2 = 1 - (SSR/SST)
                                R2[np.isinf(R2)] = np.nan; 

                                ax[i_m, 1].bar(targ + 8*tsk + 0.4*di, np.nanmean(R2), color=data_col, width = .4)
                                ax[i_m, 1].set_title('R2 Push Activity, \n %s' %mod)
    
### Generalization of neural dynamics -- population R2 ### -- figure 6(?)
def plot_r2_bar_model_7_gen(model_set_number = 7, ndays = None, use_action = False):
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    
    ########### FIGS ##########
    fco, axco = plt.subplots(ncols = 2, nrows = 2, figsize = (4, 4))
    fob, axob = plt.subplots(ncols = 2, nrows = 2, figsize = (4, 4))
    fbth, axbth = plt.subplots(ncols = 2, nrows = 2, figsize = (8, 8))

    if use_action:
        models_to_include = ['hist_1pos_0psh_1spksm_1_spksp_0']
    else:
        models_to_include = ['hist_1pos_0psh_0spksm_1_spksp_0']

    if ndays is None:
        ndays_none = True
    else:
        ndays_none = False

    for i_a, animal in enumerate(['grom', 'jeev']):
        DAYs = []; 

        RX = []; 
        RW = []; 
        RGEN = []; 

        RX_A = []; 
        RW_A = []; 
        RGEN_A = [];


        f, ax = plt.subplots(nrows = 2, ncols =2 )
        colors = ['b','r','k']
        alphas = [1.,1.,1.]

        dat = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d_action%s.pkl' %(model_set_number, use_action), 'rb'))

        if ndays_none:
            if animal == 'grom':
                ndays_all = np.arange(9);
                ndays = 9; 
            elif animal == 'jeev':
                ndays_all = [0, 2, 3];
                ndays = 3; 

        #####################
        ### Real bar plot ###
        #####################
        ### 3 plots -- one for CO, one for OB, one combined
        ### CO / OBS data  x Fit CO/OBS/GEN x ndays
        R2s_plot_spks = np.zeros((2, 3, ndays))
        R2s_plot_acts = np.zeros((2, 3, ndays)) 

        for i_d, nd in enumerate(ndays_all):

            ### Split by task -- assess R2 separately:  
            for i_t in range(2):

                ### task specific indices: 
                task_ix = np.nonzero(dat[(nd, 'task')] == i_t)[0]

                predict_key = 'spks'

                task = dat[(nd, 'task')][task_ix]
                push = dat[(nd, 'np')][task_ix, :]
                spks = dat[(nd, 'spikes')][task_ix, :]

                mn_spks = np.mean(dat[(nd, 'spikes')], axis=0)
                mn_push = np.mean(dat[(nd, 'np')], axis=0)

                if predict_key == 'spks':
                    truth = spks; 
                    truth_mn = mn_spks
                elif predict_key == 'psh':
                    truth = push; 
                    truth_mn = mn_push; 

                ### Get the KG: 
                if animal == 'grom':
                    KG, _, _ = get_KG_decoder_grom(nd)
                elif animal == 'jeev':
                    KG, _, _ = get_KG_decoder_jeev(nd)
                    KG = np.squeeze(np.array(KG))

                ### convert this to discrete commands
                commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, nd, vel_ix = [0, 1])[0]

                for i_m, mod in enumerate(models_to_include):

                    ### for different task 
                    data_mod = dat[(nd, mod)][task_ix, :, :] ### T x N x 3 [train on task 0, train on task 1, train on both]

                    for trg_ix, (tsk_col, tsk_alph) in enumerate(zip(colors, alphas)):

                        R2 = get_R2(truth, data_mod[:, :, trg_ix], pop = True, ignore_nans = True)
                        R2s_plot_spks[i_t, trg_ix, i_d] = R2; 

                        ax[i_t, 0].bar(i_d + (9*trg_ix), R2, color=tsk_col)
                        ax[i_t, 0].set_title('R2 Neural Activity, \n %s' %mod)
                        
                        if predict_key == 'spks':
                            ###### Estimated Push ######
                            R2 = get_R2(push, np.dot(KG, data_mod[:, :, trg_ix].T).T, pop = True, ignore_nans = True)
                            R2s_plot_acts[i_t, trg_ix, i_d] = R2;

                            ax[i_t, 1].bar(i_d + (9*trg_ix), R2, color=tsk_col)
                            ax[i_t, 1].set_title('R2 Push Activity, \n %s' %mod)
            

        ### Here we want to do within task vs. across task: 
        pwi = np.zeros((ndays, 2, 2)) # days  x co/obs x spks/act
        px = np.zeros((ndays, 2, 2))
        pall = np.zeros((ndays, 2, 2))

        for i_d, nd in enumerate(ndays_all):
            pred_wi = []; 
            pred_x = []; 

            ### Get the KG: 
            if animal == 'grom':
                KG, _, _ = get_KG_decoder_grom(nd)
            elif animal == 'jeev':
                KG, _, _ = get_KG_decoder_jeev(nd)
                KG = np.squeeze(np.array(KG))

            ### Data ###
            for i_t in range(2):
                task_ix = np.nonzero(dat[(nd, 'task')] == i_t)[0]
                spks_true = dat[(nd, 'spikes')][task_ix, :]
                psh_true = dat[(nd, 'np')][task_ix, :]

                ### Model used ###
                for i_train in range(3):

                    pred = dat[(nd, mod)][task_ix, :, i_train]
                    psh = np.dot(KG, pred.T).T

                    r2i = get_R2(spks_true, pred, pop = True, ignore_nans = True)
                    r2i_psh = get_R2(psh_true, psh, pop = True, ignore_nans = True)

                    if i_t == i_train:
                        pwi[i_d, i_t, 0] = r2i
                        pwi[i_d, i_t, 1] = r2i_psh

                    elif i_train == 2:
                        pall[i_d, i_t, 0] = r2i
                        pall[i_d, i_t, 1] = r2i_psh

                    else:
                        assert np.abs(i_train - i_t) == 1
                        px[i_d, i_t, 0] = r2i
                        px[i_d, i_t, 1] = r2i_psh

        PWI = [pwi[:, :, 0].reshape(-1), pwi[:, :, 1].reshape(-1), ]
        PX = [px[:, :, 0].reshape(-1), px[:, :, 1].reshape(-1) ]
        PALL = [pall[:, :, 0].reshape(-1), pall[:, :, 1].reshape(-1)]

        #### Plot figure ####
        tis = ['Neural', 'Push']
        for z in range(2):
            pwii = PWI[z]
            pxi = PX[z]
            palli = PALL[z]

            ### CO data figure ###
            # axco[z, i_a].bar(0, np.mean(R2s_plot_spks[0, 0, :]), color = 'w', edgecolor = 'k', width = .4, linewidth = 2.)
            # axco[z, i_a].errorbar(0, np.mean(R2s_plot_spks[0, 0, :]), yerr=np.std(R2s_plot_spks[0, 0, :])/np.sqrt(ndays), color = 'k', marker='|')
            # axco[z, i_a].bar(0.4, np.mean(R2s_plot_spks[0, 1, :]), color = 'grey', edgecolor = 'k', width = .4, linewidth = 2.)
            # axco[z, i_a].errorbar(0.4, np.mean(R2s_plot_spks[0, 1, :]), yerr=np.std(R2s_plot_spks[0, 1, :])/np.sqrt(ndays), color = 'k', marker='|')

            # axob[z, i_a].bar(0., np.mean(R2s_plot_spks[1, 1, :]), color = 'grey', edgecolor = 'k', width = .4, linewidth = 2.)
            # axob[z, i_a].errorbar(0, np.mean(R2s_plot_spks[1, 1, :]), yerr=np.std(R2s_plot_spks[1, 1, :])/np.sqrt(ndays), color = 'k', marker='|')
            # axob[z, i_a].bar(0.4, np.mean(R2s_plot_spks[1, 0, :]), color = 'w', edgecolor = 'k', width = .4, linewidth = 2.)
            # axob[z, i_a].errorbar(0.4, np.mean(R2s_plot_spks[1, 0, :]), yerr=np.std(R2s_plot_spks[1, 0, :])/np.sqrt(ndays), color = 'k', marker='|')

            #### Figure w/ combo ####
            axbth[z, i_a].bar(0, np.mean(pwii), color='w', edgecolor='k', width=.4, linewidth=.2)
            axbth[z, i_a].bar(0.4, np.mean(pxi), color='grey', edgecolor='k', width=.4, linewidth=.2)
            #axbth[z, i_a].bar(0.8, np.mean(palli), color='k', edgecolor='k', width=.4, linewidth=.2)
            
        
            for i_d in range(ndays):
                # axco[z, i_a].plot([0, .4], [R2s_plot_spks[0, 0, i_d], R2s_plot_spks[0, 1, i_d]], 'k-', linewidth = 1.)
                # axob[z, i_a].plot([0, .4], [R2s_plot_spks[1, 1, i_d], R2s_plot_spks[1, 0, i_d]], 'k-', linewidth = 1.)

                for x in range(2):
                    #axbth[z, i_a].plot([0, .4, .8], [pwii[2*i_d + x], pxi[2*i_d + x], palli[2*i_d + x]], 'k-', linewidth = 1.)
                    axbth[z, i_a].plot([0, .4,], [pwii[2*i_d + x], pxi[2*i_d + x]], 'k-', linewidth = 1.)

                    DAYs.append(i_d)

                    if z == 0:
                        RX.append(px[i_d, x, 0])
                        RW.append(pwi[i_d, x, 0])
                        RGEN.append(pall[i_d, x, 0])

                    elif z == 1:
                        RX_A.append(px[i_d, x, 1])
                        RW_A.append(pwi[i_d, x, 1])
                        RGEN_A.append(pall[i_d, x, 1])


            axbth[z, i_a].set_title('%s, Monk %s'%(tis[z], animal),fontsize = 8)
            axbth[z, i_a].set_ylabel('R2')
            axbth[z, i_a].set_xticks([0., .4])
            axbth[z, i_a].set_xticklabels(['Within', 'Across'])
        ### stats: 

        print 'Neural, subj %s, w vs x' %(animal)
        w_vs_x = np.hstack(( np.hstack((RW)), np.hstack((RX)) ))
        grps = np.hstack(( np.zeros_like(RW), np.zeros_like(RX)+1))
        pv, slp = run_LME(DAYs, grps, w_vs_x)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '***')
        elif pv < 0.01: 
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '**')
        elif pv < 0.05:
            axbth[0, i_a].text(0.2, np.max(w_vs_x), '*')


        print 'ACTION, subj %s, w vs x' %(animal)
        w_vs_x = np.hstack(( np.hstack((RW_A)), np.hstack((RX_A)) ))
        grps = np.hstack(( np.zeros_like(RW_A), np.zeros_like(RX_A)+1))
        pv, slp = run_LME(DAYs, grps, w_vs_x)
        print('pv: %.2f, slp %.2f'%(pv, slp))

        if pv < 0.001: 
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '***')
        elif pv < 0.01: 
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '**')
        elif pv < 0.05:
            axbth[1, i_a].text(0.2, np.max(w_vs_x), '*')
        else:
            axbth[1, i_a].text(0.2, np.max(w_vs_x), 'n.s.')



    fco.tight_layout()
    fob.tight_layout()
    fbth.tight_layout()
    fbth.savefig('gen_w_vs_x.svg')

def plot_yt_given_st(animal='grom', model_set_number = 8, min_obs = 15):
    model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %(model_set_number), 'rb'))
    mod_state1     = 'prespos_1psh_0spksm_0_spksp_0'
    mod_state2     = 'prespos_-1psh_0spksm_0_spksp_0'
    i_d = 0; 

    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    tsk  = model_dict[i_d, 'task']
    targ = model_dict[i_d, 'trg']
    push = model_dict[i_d, 'np']
    commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]


    plot_ylab = ['y_t', 'y_t|s_t', 'y_t - y_t|s_t', 'y_t_mean_all', 'y_t_mean_command']

    f, ax = plt.subplots(ncols = 5, nrows=2, figsize=(15, 8))
    for i_m, mod_state in enumerate([mod_state1, mod_state2]):

        pred_y = model_dict[i_d, mod_state]
        true_y = model_dict[i_d, 'spks']

        ### Residual: 
        res_y = true_y - pred_y; 

        T = model_dict[i_d, mod_state]
        mean_y = np.zeros_like(true_y) + np.mean(true_y, axis=0)[np.newaxis, :]

        command_y_mn = np.zeros_like(true_y)
        for m in range(4):
            for a in range(8):
                ix = np.nonzero( (commands_disc[:, 0] == m) & (commands_disc[:, 1] == a) == True)[0]
                command_y_mn[ix, :] = np.mean(true_y[ix, :], axis=0)

        X = dict(); 
        Y = dict(); 

        for i in range(5):
            X[i] = []; 
            Y[i] = []; 

        for t in range(2):
            for g in range(8):

                ### Command: 
                for m in range(4):
                    for a in range(8):

                        ### Get indices; 
                        ix = (tsk == t) & (targ == g) & (commands_disc[:, 0] == m) & (commands_disc[:, 1] == a)
                        ix = np.nonzero(ix == True)[0]

                        if len(ix) >= min_obs:
                            ### Take means: 
                            for ia, act in enumerate([true_y, pred_y, res_y, mean_y, command_y_mn]):
                                ax[i_m, ia].plot(np.mean(true_y[ix, :], axis=0), np.mean(act[ix, :], axis=0),'.', markersize=1.5)
                                Y[ia].append(np.mean(act[ix, :], axis=0))
                                X[ia].append(np.mean(true_y[ix, :], axis=0))

        ### At the end do these correlations; 
        for i, nm in enumerate(plot_ylab):
            x = np.hstack((X[i]))
            y = np.hstack((Y[i]))

            ix = np.nonzero(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
            
            slp,intc,rv,pv,err =scipy.stats.linregress(x[ix], y[ix])
            x_ = np.linspace(np.nanmin(x), np.nanmax(x), 100.)
            y_ = slp*x_ + intc; 

            ax[i_m, i].plot(x_, y_, '-', color='gray', linewidth=1.)
            ax[i_m, i].set_title(nm+', rv: %.2f, slp: %.2f' %(rv, slp))

### Plot real mean diffs across task/target -- figure 2, 2019
def plot_real_mean_diffs(model_set_number = 3, min_obs = 15, plot_ex = False, plot_disc = False, cov = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
            width = 3.
        elif animal == 'jeev':
            ndays = 4; 
            width = 2.

        ### Now bar plot that: 
        f, ax = plt.subplots(figsize=(width, 4))
        
        for i_d in range(ndays):
            if plot_ex: 
                fex, axex = plt.subplots(ncols = 10, nrows = 5, figsize = (12, 6))
                axex_cnt = 0; 

            diffs = []; diffs_cov = []; 
            sig_diffs = []; 

            spks = model_dict[i_d, 'spks']
            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']

            commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ##################################
            ######### SNR / DISC THING #######
            #################################
            #min_max = np.zeros((N, 2, 2))
            #snr = np.zeros_like(min_max)

            if plot_disc:
                n_disc = np.zeros((4, 8, 2))
                ### For each neuron, get the min/max mean command for a command/task/target: 

                if i_d == 0: 
                    for tsk_i in range(2): 
                        if tsk_i == 0:
                            targeti = 4; 
                        elif tsk_i == 1:
                            targeti = 5; 
                        for mag_i in range(4):
                            for ang_i in range(8): 
                                ix = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == tsk_i) & (targ == targeti)
                                #if len(np.nonzero(ix==True)[0]) > min_obs:
                                ix = np.nonzero(ix == True)[0]
                                n = 38; 
                                if len(ix) > 0:
                                    n_disc[mag_i, ang_i, tsk_i] = np.nanmean(spks[ix, n])

                    ### Make a disc plot: 
                    disc_plot(n_disc)

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:
                            
                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ### make the plot: 
                                    if cov:
                                        diffs = get_cov_diffs(ix_co, ix_ob, spks, diffs)
                                    else:
                                        diffs.append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))

                                    #### Get the covariance and plot that: #####

                                    for n in range(N): 
                                        ks, pv = scipy.stats.ks_2samp(spks[ix_co, n], spks[ix_ob, n])
                                        if pv < 0.05:
                                            sig_diffs.append(np.mean(spks[ix_co, n]) - np.mean(spks[ix_ob, n]))

                                        if plot_ex:
                                            #if np.logical_and(len(ix_co) > 30, len(ix_ob) > 30):
                                            if axex_cnt < 40:
                                                if np.logical_and(mag_i > 1, n == 38): 
                                                    axi = axex[axex_cnt / 9, axex_cnt %9]

                                                    draw_plot(0, 10*spks[ix_co, n], co_obs_cmap[0], 'white', axi)
                                                    draw_plot(1, 10*spks[ix_ob, n], co_obs_cmap[1], 'white', axi)

                                                    axi.set_title('A:%d, M:%d, NN%d, \nCOT: %d, OBST: %d, Monk:%s'%(ang_i, mag_i, n, targi, targi2, animal),fontsize=6)
                                                    axi.set_ylabel('Firing Rate (Hz)')
                                                    axi.set_xlim([-.5, 1.5])
                                                    #axi.set_ylim([-1., 10*(1+np.max(np.hstack(( spks[ix_co, n], spks[ix_ob, n]))))])
                                                    axex_cnt += 1
                                                    if axex_cnt == 40:
                                                        fex.tight_layout()
                                                        fex.savefig(fig_dir + 'monk_%s_ex_mean_diffs.svg'%animal)

            if len(sig_diffs) > 0:
                SD = np.abs(np.hstack((sig_diffs)))*10 # to Hz
            else:
                SD = []; 
            AD = np.abs(np.hstack((diffs)))*10 # to Hz

            for i, (D, col) in enumerate(zip([AD], ['k'])): #, SD, CD], ['k', 'r', 'b'])):
                draw_plot(i_d, D, 'k', [1, 1, 1, .2], ax, .8)
                #ax.bar(i_d + i*.3, np.mean(D), color='w', edgecolor='k', width=.8, linewidth=1.)
                #ax.errorbar(i_d + i*.3, np.mean(D), np.std(D)/np.sqrt(len(D)), marker='|', color=col)
                ax.plot(np.random.randn(len(D))*.1 + i*.3 + i_d, D, '.', color='gray', markersize=1.5, alpha = .5)
        
        ax.set_ylabel('Abs Hz Diff Across Conditions')
        ax.set_xlabel('Days')
        ax.set_xlim([-1, ndays + 1])
        if cov:
            pass
        else:
            ax.set_ylim([0., 20.])
        ax.set_title('Monk '+animal[0].capitalize())
        f.tight_layout()
        f.savefig(fig_dir + 'monk_%s_mean_diffs_box_plots_cov_%s.png' %(animal, str(cov)), transparent = True)

        ### Need to stats test the differences across populations: 

def plot_real_mean_diffs_wi_vs_x(model_set_number = 3, min_obs = 15, cov = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now bar plot that: 
    fsumm, axsumm = plt.subplots(figsize=(4, 4))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
        elif animal == 'jeev':
            ndays = 4; 

        DWI = []; 
        DX = []; 
        days = []; mets = []; grp = []; 
        mnz = np.zeros((ndays, 2)) # days x [x / wi]

        f, ax = plt.subplots(figsize=(ndays/2., 4))

        for i_d in range(ndays):

            diffs_wi = []; 
            diffs_x = []; 

            spks = model_dict[i_d, 'spks']
            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']

            commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:

                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    ix_ob0 = np.nonzero(ix_ob == True)[0]

                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co2, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob2, :], axis=0))) == 0
                                    
                                    if cov:
                                        ### make the plot: 
                                        diffs_wi = get_cov_diffs(ix_co1, ix_co2, spks, diffs_wi)
                                        diffs_wi = get_cov_diffs(ix_ob1, ix_ob2, spks, diffs_wi)

                                        diffs_x = get_cov_diffs(ix_co1, ix_ob1, spks, diffs_x)
                                        diffs_x = get_cov_diffs(ix_co2, ix_ob2, spks, diffs_x)

                                    else:
                                        ### make the plot: 
                                        diffs_wi.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_co2, :], axis=0))
                                        diffs_wi.append(np.mean(spks[ix_ob1, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

                                        diffs_x.append(np.mean(spks[ix_co1, :], axis=0) - np.mean(spks[ix_ob1, :], axis=0))
                                        diffs_x.append(np.mean(spks[ix_co2, :], axis=0) - np.mean(spks[ix_ob2, :], axis=0))

            if cov:
                mult = 1; 
            else: 
                mult = 10; 
            AD_wi = np.abs(np.hstack((diffs_wi)))*mult # to Hz
            AD_x = np.abs(np.hstack((diffs_x)))*mult # To hz

            DWI.append(AD_wi)
            DX.append(AD_x)

            days.append(np.hstack(( np.zeros_like(AD_wi) + i_d, np.zeros_like(AD_x) + i_d)))
            grp.append( np.hstack(( np.zeros_like(AD_wi) + 1,   np.zeros_like(AD_x)))) # looking for + slp
            mets.append(AD_wi)
            mets.append(AD_x)

            for i, (D, col) in enumerate(zip([AD_x, AD_wi], ['k', 'gray'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color=col, edgecolor='none', width=.4, linewidth=1.0)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        DWI = np.hstack((DWI))
        DX = np.hstack((DX))
        days = np.hstack((days))
        grp = np.hstack((grp))
        mets = np.hstack((mets))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = run_LME(days, grp, mets)

        print 'LME model, fixed effect is day, rand effect is X vs. Wi., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ###
        axsumm.bar(0 + ia, np.mean(DX), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha = .8)
        axsumm.bar(0.4 + ia, np.mean(DWI), color='gray', edgecolor='none', width=.4, linewidth=2.0, alpha =.8)

        for i_d in range(ndays):
            axsumm.plot(np.array([0, .4]) + ia, mnz[i_d, :], '-', color='k', linewidth=1.0)

        if pv < 0.001: 
            axsumm.text(0.2+ia, np.max(mnz), '***')
        elif pv < 0.01: 
            axsumm.text(0.2+ia, np.max(mnz), '**')
        elif pv < 0.05: 
            axsumm.text(0.2+ia, np.max(mnz), '*')
        else:
            axsumm.text(0.2+ia, np.max(mnz), 'n.s.')

        # ax.set_ylabel('Difference in Hz')
        # ax.set_xlabel('Days')
        # ax.set_xticks(np.arange(ndays))
        # ax.set_title('Monk '+animal[0].capitalize())
        # f.tight_layout()

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J']) 
    if cov:
        # axsumm.set_ylim([0, 30])
        # axsumm.set_ylabel(' Cov Diffs ($Hz^2$) ') 
        axsumm.set_ylim([0, .6])
        axsumm.set_ylabel(' Main Cov. Overlap ') 
    else:
        axsumm.set_ylim([0, 5])
        axsumm.set_ylabel(' Mean Diffs (Hz) ') 
    fsumm.tight_layout()
    fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_mean_diffs_cov%s.svg'%(str(cov)))

### Figure 1 -- null activyt 
def plot_real_mean_diffs_x_null_vs_potent(model_set_number = 3, min_obs = 15, cov = False):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones
    ### Plot cov. diffs (mat1 - mat2)
    
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Now bar plot that: 
    fsumm, axsumm = plt.subplots(figsize=(4, 4))
    NP = dict()

    for ia, animal in enumerate(['grom','jeev']):
        NP[animal, 'pot'] = []
        NP[animal, 'nul'] = []
        NP[animal, 'tot'] = []

        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
        elif animal == 'jeev':
            ndays = 4; 

        DX_pot = []; 
        DX_nul = []; 
        days = []; grp = []; mets = [];
        mnz = np.zeros((ndays, 2)) # days x [pot / nul]

        f, ax = plt.subplots(figsize=(ndays/2., 4))

        for i_d in range(ndays):

            diffs_null = []; 
            diffs_pot = []; 

            spks = model_dict[i_d, 'spks']

            ### Get the decoder ###
            ### Get kalman gain etc. 
            if animal == 'grom':
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_grom(i_d)

            elif animal == 'jeev':
                KG, KG_null_proj, KG_potent_orth = get_KG_decoder_jeev(i_d)

            ### Get null and potent -- KG_null_proj, KG_potent_orth
            spks_null = np.dot(KG_null_proj, spks.T).T
            spks_pot =  np.dot(KG_potent_orth, spks.T).T

            NP[animal, 'nul'].append(np.var(10*spks_null, axis=0))
            NP[animal, 'pot'].append(np.var(10*spks_pot, axis=0))
            NP[animal, 'tot'].append(np.var(10*spks, axis=0))

            assert np.allclose(spks, spks_null + spks_pot)

            N = spks.shape[1]; 
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']

            commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:

                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    ix_ob0 = np.nonzero(ix_ob == True)[0]

                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_co2, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob1, :], axis=0))) == 0
                                    assert np.sum(np.isnan(np.mean(spks[ix_ob2, :], axis=0))) == 0
                                    
                                    ### make the plot: 
                                    if cov: 
                                        diffs_null = get_cov_diffs(ix_co1, ix_ob1, spks_null, diffs_null)
                                        diffs_null = get_cov_diffs(ix_co2, ix_ob2, spks_null, diffs_null)

                                        diffs_pot = get_cov_diffs(ix_co1, ix_ob1, spks_pot, diffs_pot)
                                        diffs_pot = get_cov_diffs(ix_co2, ix_ob2, spks_pot, diffs_pot)

                                    else:
                                        diffs_null.append(np.mean(spks_null[ix_co1, :], axis=0) - np.mean(spks_null[ix_ob1, :], axis=0))
                                        diffs_null.append(np.mean(spks_null[ix_co2, :], axis=0) - np.mean(spks_null[ix_ob2, :], axis=0))

                                        diffs_pot.append(np.mean(spks_pot[ix_co1, :], axis=0) - np.mean(spks_pot[ix_ob1, :], axis=0))
                                        diffs_pot.append(np.mean(spks_pot[ix_co2, :], axis=0) - np.mean(spks_pot[ix_ob2, :], axis=0))

            if cov: 
                mult = 1; 
            else:
                mult = 10; 

            AD_nul = np.abs(np.hstack((diffs_null)))*mult # to Hz
            AD_pot = np.abs(np.hstack((diffs_pot)))*mult # To hz

            DX_nul.append(AD_nul)
            DX_pot.append(AD_pot)
            days.append(np.hstack(( np.zeros_like(AD_nul) + i_d, np.zeros_like(AD_pot) + i_d)))
            grp.append(np.zeros_like(AD_nul))
            grp.append(np.ones_like(AD_pot))
            mets.append(AD_nul)
            mets.append(AD_pot)

            for i, (D, col) in enumerate(zip([AD_nul, AD_pot], ['r', 'k'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color='k', edgecolor=col, width=.4, linewidth=1.0, alpha=.8)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        grp = np.hstack((grp))
        mets = np.hstack((mets))
        days = np.hstack((days))

        DX_nul = np.hstack((DX_nul))
        DX_pot = np.hstack((DX_pot))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = run_LME(days, grp, mets)

        print 'LME model, fixed effect is day, rand effect is nul vs. pot., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ##
        axsumm.bar(0 + ia, np.mean(DX_nul), color='k', edgecolor='r', width=.4, linewidth=2.0, alpha = .7,)
        axsumm.bar(0.4 + ia, np.mean(DX_pot), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha =.7)

        for i_d in range(ndays):
            axsumm.plot(np.array([0, .4]) + ia, mnz[i_d, :], '-', color='k', linewidth=1.0)

        if pv < 0.001: 
            axsumm.text(0.2+ia, np.max(mnz), '***')
        elif pv < 0.01: 
            axsumm.text(0.2+ia, np.max(mnz), '**')
        elif pv < 0.05: 
            axsumm.text(0.2+ia, np.max(mnz), '*')
        else:
            axsumm.text(0.2+ia, np.max(mnz), 'n.s.')

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J'])
    if cov:
        # axsumm.set_ylabel(' Abs Cov Diffs ($Hz^2$) ')  
        # axsumm.set_ylim([0, 30])
        axsumm.set_ylim([0., 6.])
        axsumm.set_ylabel(['Main Cov. Overlap'])

    else:
        axsumm.set_ylabel(' Abs Mean Diffs (Hz) ')  
        axsumm.set_ylim([0, 5])
    fsumm.tight_layout()
    #fsumm.savefig(fig_dir+'both_monks_nul_vs_pot_x_task_mean_diffs_cov%s.svg'%(str(cov)))

    ### Figure for nul variance; -- each list in NP[animal, 'nul'] is an Nx1 array of neural variance for that day (null or pot)
    fnul, axnul = plt.subplots(figsize = (4, 4))
    for ia, animal in enumerate(['grom', 'jeev']):
        M = []; D = []; O = []; 
        nul = np.hstack((NP[animal, 'nul']))
        pot = np.hstack((NP[animal, 'pot']))

        axnul.bar(ia, np.mean(nul), width=.4, color='k', edgecolor='r', linewidth=1.5, alpha=0.8)
        axnul.bar(ia + .41, np.mean(pot), width=.4, color='k', edgecolor='none', linewidth=1.5, alpha = 0.8)

        ymax = 0; 

        for i_d in range(len(NP[animal, 'nul'])):
            nul = NP[animal, 'nul'][i_d]
            pot = NP[animal, 'pot'][i_d]

            M.append(nul)
            M.append(pot)

            D.append(np.zeros_like(nul) + i_d)
            D.append(np.zeros_like(pot) + i_d)

            O.append(np.zeros_like(nul) + 0)
            O.append(np.zeros_like(pot) + 1)

            axnul.plot([ia, ia+.41], [np.mean(nul), np.mean(pot)], 'k-')
            ymax = np.max([ymax, np.mean(nul)])
            ymax = np.max([ymax, np.mean(pot)])

        M = np.hstack((M))
        D = np.hstack((D))
        O = np.hstack((O))

        ### Plots: 
        pv, slp = run_LME(D, O, M)

        if pv < 0.001:
            axnul.text(ia+.2, ymax, '***')
        elif pv < 0.01:
            axnul.text(ia+.2, ymax, '**')
        if pv < 0.05:
            axnul.text(ia+.2, ymax, '*')

    axnul.set_xticks([.4, 1.4])
    axnul.set_xticklabels(['G', 'J'])
    axnul.set_ylabel('Neural Variance ($Hz^2$)')
    axnul.set_ylim([0., 175.])
    fnul.tight_layout()
    #fnul.savefig(fig_dir+'null_pot_neural_var.svg')

### Figure 5 -- next behavior differences: 
def plot_real_mean_diffs_behavior_next(model_set_number = 3, min_obs = 15):
    ### Take real task / target / command / neuron / day comparisons for each neuron in the BMI
    ### Plot within a bar 
    ### Plot only sig. different ones

    ### Plot cov. diffs (mat1 - mat2)
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))
    fsumm, axsumm = plt.subplots(figsize=(4, 4))

    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
            width = 3.
        elif animal == 'jeev':
            ndays = 4; 
            width = 2.

        ### Now bar plot that: 
        f, ax = plt.subplots(figsize=(width, 4))
        
        days = []; DX = []; DWI = []; 
        cat = []; 
        met = []; 
        mnz = np.zeros((ndays, 2))

        for i_d in range(ndays):

            diffs_x = []; diffs_wi = [];
            tsk  = model_dict[i_d, 'task']
            targ = model_dict[i_d, 'trg']
            push = model_dict[i_d, 'np']
            bin_num = model_dict[i_d, 'bin_num']
            min_bin = np.min(bin_num)
            print('min bin: %d'%min_bin)

            commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

            ### Now go through combos and plot 
            for mag_i in range(4):
                for ang_i in range(8):
                    for targi in range(8):
                        ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                        
                        if len(np.nonzero(ix_co == True)[0]) > min_obs:
                            
                            for targi2 in range(8):
                                ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 1) & (targ == targi2)

                                if len(np.nonzero(ix_ob == True)[0]) > min_obs:
                                    
                                    ix_co0 = np.nonzero(ix_co == True)[0]
                                    ix_ob0 = np.nonzero(ix_ob == True)[0]

                                    ### Get the NEXT COMMAND ###
                                    ix_co0 = ix_co0 + 1; 
                                    ix_ob0 = ix_ob0 + 1; 

                                    ### Get rid of bins that aren't from prev time step
                                    ix_co0 = ix_co0[ix_co0 < len(push)]
                                    ix_co_keep = np.nonzero(bin_num[ix_co0] > min_bin)[0]
                                    ix_co0 = ix_co0[ix_co_keep]

                                    ix_ob0 = ix_ob0[ix_ob0 < len(push)]
                                    ix_ob_keep = np.nonzero(bin_num[ix_ob0] > min_bin)[0]
                                    ix_ob0 = ix_ob0[ix_ob_keep]

                                    ii = np.random.permutation(len(ix_co0))
                                    i1 = ii[:int(len(ix_co0)/2.)]
                                    i2 = ii[int(len(ix_co0)/2.):]

                                    jj = np.random.permutation(len(ix_ob0))
                                    j1 = jj[:int(len(ix_ob0)/2.)]
                                    j2 = jj[int(len(ix_ob0)/2.):]

                                    ix_co1 = ix_co0[i1]
                                    ix_co2 = ix_co0[i2]
                                    ix_ob1 = ix_ob0[j1]
                                    ix_ob2 = ix_ob0[j2]

                                    assert np.sum(np.isnan(ix_co1)) == np.sum(np.isnan(ix_co2)) == np.sum(np.isnan(ix_ob1)) == np.sum(np.isnan(ix_ob2)) == 0
                                
                                    ### make the plot: 
                                    diffs_wi.append(np.mean(push[ix_co1, :], axis=0) - np.mean(push[ix_co2, :], axis=0))
                                    diffs_wi.append(np.mean(push[ix_ob1, :], axis=0) - np.mean(push[ix_ob2, :], axis=0))

                                    diffs_x.append(np.mean(push[ix_co1, :], axis=0) - np.mean(push[ix_ob1, :], axis=0))
                                    diffs_x.append(np.mean(push[ix_co2, :], axis=0) - np.mean(push[ix_ob2, :], axis=0))

            W = np.abs(np.hstack((diffs_wi))) # to Hz
            X = np.abs(np.hstack((diffs_x))) 
            DX.append(X); 
            DWI.append(W); 
            tmp = np.hstack((W, X))
            days.append(np.zeros_like(tmp) + i_d)
            cat.append(np.zeros_like(W))
            cat.append(np.zeros_like(X) + 1)
            met.append(tmp)

            for i, (D, col) in enumerate(zip([X, W], ['k', 'gray'])):
                ax.bar(i_d + i*.45, np.nanmean(D), color=col, edgecolor='none', width=.4, linewidth=1.0)
                ax.errorbar(i_d + i*.45, np.nanmean(D), np.nanstd(D)/np.sqrt(len(D)), marker='|', color=col)
                mnz[i_d, i] = np.nanmean(D)
         
        days = np.hstack((days))
        cat = np.hstack((cat))
        met = np.hstack((met))

        ### look at: run_LME(Days, Grp, Metric):
        pv, slp = run_LME(days, cat, met)

        print 'LME model, fixed effect is day, rand effect is X vs. Wi., N = %d, ndays = %d, pv = %.4f, slp = %.4f' %(len(days), len(np.unique(days)), pv, slp)

        ###
        DX = np.hstack((DX))
        DWI = np.hstack((DWI))

        axsumm.bar(0 + ia, np.mean(DX), color='k', edgecolor='none', width=.4, linewidth=2.0, alpha = .8)
        axsumm.bar(0.4 + ia, np.mean(DWI), color='gray', edgecolor='none', width=.4, linewidth=2.0, alpha =.8)

        for i_d in range(ndays):
            axsumm.plot(np.array([0, .4]) + ia, mnz[i_d, :], '-', color='k', linewidth=1.0)

        if pv < 0.001: 
            axsumm.text(0.2+ia, np.max(mnz), '***')
        elif pv < 0.01: 
            axsumm.text(0.2+ia, np.max(mnz), '**')
        elif pv < 0.05: 
            axsumm.text(0.2+ia, np.max(mnz), '*')
        else:
            axsumm.text(0.2+ia, np.max(mnz), 'n.s.')

        # ax.set_ylabel('Difference in Hz')
        # ax.set_xlabel('Days')
        # ax.set_xticks(np.arange(ndays))
        # ax.set_title('Monk '+animal[0].capitalize())
        # f.tight_layout()

        ### Need to stats test the differences across populations:    
    axsumm.set_xticks([0.2, 1.2])
    axsumm.set_xticklabels(['G', 'J']) 
    #axsumm.set_ylim([0, 5])
    axsumm.set_ylabel(' Mean Diffs (cm/sec) ') 
    fsumm.tight_layout()
    fsumm.savefig(fig_dir+'both_monks_w_vs_x_task_push_mean_diffs.svg')

def draw_plot(xax, data, edge_color, fill_color, ax, width = .5):
    bp = ax.boxplot(data, patch_artist=True, positions = [xax], widths=[width])

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 

### Bar R2 and correlation plots -- figure 4;
def plot_real_vs_pred(model_set_number = 1, min_obs = 15, cov = True):
    
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    models_to_include = ['prespos_0psh_1spksm_0_spksp_0', 
                         'hist_1pos_3psh_1spksm_0_spksp_0', 
                         'hist_1pos_3psh_1spksm_1_spksp_0']

    models_to_include_labs = ['y_t | a_t', 
                              'y_t | a_t, s_{t-1}, s_tFinal, tsk', 
                              'y_t | a_t, s_{t-1}, y_{t-1}']


    for ia, animal in enumerate(['grom','jeev']):
        model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))

        if animal == 'grom':
            ndays = 9; 
        elif animal == 'jeev':
            ndays = 4; 

        diffs = dict(); 
        sig_diffs = dict(); 

        MOD = []; DAY = []; VAL = []; 

        pred_diffs = dict(); 
        pred_sig_diffs = dict(); 

        R2 = dict(); NN = dict(); 

        fbar, axbar = plt.subplots(figsize=(3, 4))
        fall, axall = plt.subplots(ncols = len(models_to_include), nrows = ndays, figsize=(10, 10))

        for i_m, model in enumerate(models_to_include):

            ### For each model, setup in teh diffs 
            diffs[model] = []
            sig_diffs[model] = []
            pred_diffs[model] = []
            pred_sig_diffs[model] = []

            for i_d in range(ndays):
                
                R2[model, i_d] = []; 
                ### Get the spiking data
                spks = model_dict[i_d, 'spks']
                pred = model_dict[i_d, model]

                ### Get the task parameters
                tsk  = model_dict[i_d, 'task']
                targ = model_dict[i_d, 'trg']
                push = model_dict[i_d, 'np']
                
                ### Setup the dicitonaries to be 
                diffs[model, i_d] = []
                sig_diffs[model, i_d] = []
                pred_diffs[model, i_d] = []
                pred_sig_diffs[model, i_d] = []

                ### Get the discretized commands
                commands_disc = subspace_overlap.commands2bins([push], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]

                ### Now go through combos and plot 
                for mag_i in range(4):
                    for ang_i in range(8):
                        for targi in range(8):
                            ### Get co / task 
                            ix_co = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi)
                            ix_co = np.nonzero(ix_co == True)[0]

                            if len(ix_co) >= min_obs:
                                
                                ### Get info second task: 
                                for targi2 in range(8):
                                    ix_ob = (commands_disc[:, 0] == mag_i) & (commands_disc[:, 1] == ang_i) & (tsk == 0) & (targ == targi2)
                                    ix_ob = np.nonzero(ix_ob == True)[0]

                                    if len(ix_ob) >= min_obs:

                                        assert(len(ix_co) >= min_obs)
                                        assert(len(ix_ob) >= min_obs)
                                        #print 'adding: mag_i: %d, ang_i: %d, targi: %d, targi2: %d' %(mag_i, ang_i, targi, targi2)
                                        ### make the plot: 
                                        if cov: 
                                            diffs[model, i_d] = get_cov_diffs(ix_co, ix_ob, spks, diffs[model, i_d])
                                            pred_diffs[model, i_d] = get_cov_diffs(ix_co, ix_ob, pred, pred_diffs[model, i_d])
                                        else: 
                                            diffs[model, i_d].append(np.mean(spks[ix_co, :], axis=0) - np.mean(spks[ix_ob, :], axis=0))    
                                            pred_diffs[model, i_d].append(np.mean(pred[ix_co, :], axis=0) - np.mean(pred[ix_ob, :], axis=0))


            ### Now scatter plot all data over all days: 
            #f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))

            ### Make R2 plots to show how much each plot accounts for variace: 
            for i_d in range(ndays):

                ### T x N
                x = np.vstack((diffs[model, i_d]))
                y = np.vstack((pred_diffs[model, i_d]))
                
                #axall[i_d, i_m].plot(x, y, '.', markersize=2.)
                ### get variance explained -- here, each point is a neuron / command / day / targ1 / targ 2 difference
                ### the mean for SST is the neuron specific avg. true difference. 
                VAF = get_R2(x, y, pop = True)

                ### Old VAF: 
                #VAF = 1 - np.sum((x-y)**2)/np.sum((x-np.mean(x))**2)

                R2[model, i_d].append(VAF);
                axall[i_d, i_m].set_title(VAF, fontsize=6)

                MOD.append(i_m)
                DAY.append(i_d)
                VAL.append(VAF)

        fall.tight_layout()
        MOD = np.hstack((MOD))
        DAY = np.hstack((DAY))
        VAL = np.hstack((VAL))

        ### Plot indices ###
        ix0 = np.nonzero(MOD < 2)[0]
        ix1 = np.nonzero(MOD > 0)[0]

        pv0, slp0 = run_LME(DAY[ix0], MOD[ix0], VAL[ix0])
        pv1, slp1 = run_LME(DAY[ix1], MOD[ix1], VAL[ix1])

        print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix0])), pv0, slp0, len(ix0)))
        print('Animal %s, Mods %s, pv: %.3f, slp: %.3f, N: %d' %(animal, str(np.unique(MOD[ix1])), pv1, slp1, len(ix1)))

        ### Plot as bar plot ###
        all_data = {}
        for i_m, model in enumerate(models_to_include):
            all_data[model] = []; 

        for i_d in range(ndays):
            tmp = []; 

            for i_m, model in enumerate(models_to_include):
                r2 = np.hstack((R2[model, i_d]))
                r2[np.isinf(r2)] = np.nan
                tmp.append(np.nanmean(r2))
                all_data[model].append(r2)
            axbar.plot(np.arange(len(models_to_include)), tmp, '-', color='gray')

        for i_m, model in enumerate(models_to_include):
            tmp2 = np.hstack((all_data[model]))
            tmp2 = tmp2[~np.isnan(tmp2)]
            axbar.bar(i_m, np.mean(tmp2), color = model_cols[i_m], edgecolor='k', linewidth=2.)
            axbar.errorbar(i_m, np.mean(tmp2), yerr=np.std(tmp2)/np.sqrt(len(tmp2)), color = 'k', marker='|')

        axbar.set_xticks(np.arange(len(models_to_include)))
        models_to_include_labs_tex = ['$' + m + '$' for m in models_to_include_labs]
        axbar.set_xticklabels(models_to_include_labs_tex, rotation = 45, fontsize=10)
        fbar.tight_layout()
        fbar.savefig('/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/monk_%s_r2_comparison_mean_diffs_cov%s.svg' %(animal, str(cov)))
        
def print_num_trials_per_day_CO_OBS():
    perc = np.zeros((2, 10, 2))

    for i_a, animal in enumerate(['grom', 'jeev']):
        if animal == 'grom':
            order_dict = co_obs_tuning_matrices.ordered_input_type
            input_type = analysis_config.data_params['grom_input_type']

        elif animal == 'jeev':
            order_dict = file_key.ordered_task_filelist
            input_type = file_key.task_filelist

        for i_d, day in enumerate(input_type):

            data, data_temp, spikes, sub_spk_temp_all, sub_push_all = get_spike_kinematics(animal,
                day, order_dict[i_d], 1)

            T = len(data_temp)
            co_ix = np.nonzero(data_temp['tsk'] == 0)[0]
            ob_ix = np.nonzero(data_temp['tsk'] == 1)[0]

            perc[i_a, i_d, 0] = len(co_ix) / float(T)
            perc[i_a, i_d, 1] = len(ob_ix) / float(T)
    return perc

def get_perc_sig_model(hdf, model, unit_ix, ignore_vars=None):
    '''
    Summary: method to return how many units have a significantly tuned parameters 
        -- helper fcn for 'plot_hdf_model_results'
    Input param: hdf: hdf table
    Input param: model: model name
    Input param: ignore_vars: list of variables to ignore (e.g. ['Intercept'])
    Output param: 
    '''
    mod = getattr(hdf.root, model)
    mod_vars = getattr(hdf.root, model+'_nms')

    if ignore_vars is None:
        ignore_vars = ['']

    # Indices of variables we care about: 
    ix = [i for i, j in enumerate(mod_vars.vars[:]) if j not in ignore_vars]
    print 'Using variables: ', 
    print mod_vars.vars[ix]
    # PValues: 
    pv = mod[unit_ix]['pvalue'][:, ix]

    # Number of times significant: 
    sm = np.sum(pv < 0.05, axis=1)

    # Indices of units with significant modulation: 
    sm0 = np.nonzero(sm)[0]
    return sm0

def plot_hdf_model_results(filenames, animals = ['grom', 'jeev']):
    '''
    Summary: plot bar plots for grom & jeev for various tuning metrics
    Input param: filenames: list with HDF filenames for grom, jeev. Usually: 
        ['/Volumes/TimeMachineBackups/grom2016/grom_unit_hdf_file_trl_tsk_v2.h5','/Volumes/TimeMachineBackups/jeev2013/jeev_unit_hdf_file_trl_tsk_v2.h5']
    
    Output param: various plots!
    '''

    for ia, (fname, animal) in enumerate(zip(filenames, animals)):
        hdf = tables.openFile(fname)

        ### Percent w/ sig task-variable (as offset) ###
        model_types = ['v', 'pv', 'pva', 'p', 'vh', 'phvh', 'ph', 'ph1', 'phvh1']
        
        perc_tsk_mod_of_sig_cells = {}
        perc_tsk_mod_of_sig_cells_w_trl = {}
        r2 = {}
        aic = {}
        bic = {}

        day = getattr(hdf.root, model_types[0])
        day_ixs = np.squeeze(day[:]['day_ix'])
        n_day = np.unique(day_ixs)

        for i_d, dy in enumerate(n_day):
            perc_tsk_mod_of_sig_cells[dy] = []
            perc_tsk_mod_of_sig_cells_w_trl[dy] = []
            r2[dy] = []
            aic[dy] = []
            bic[dy] = []

            # Day indices: 
            day_ix = np.nonzero(day_ixs==dy)[0]

            # Iterate through the model types: 
            for im, m in enumerate(model_types):
                
                # Percent task modulated: 
                stat = 0
                try:
                    sig_tuned_ix = get_perc_sig_model(hdf, m, day_ix, ignore_vars=['Intercept'])
                    sig_tuned_ix_w_tsk = get_perc_sig_model(hdf, m+'_tsk', day_ix, ignore_vars=['Intercept'])
                    perc_tsk_mod_of_sig_cells[dy].append(len(np.intersect1d(sig_tuned_ix, sig_tuned_ix_w_tsk))/float(len(sig_tuned_ix)))
                    stat += 1
                    # Percent task modulated when trl is included in the beginning: 
                    sig_tuned_ix_trl = get_perc_sig_model(hdf, m+'_trl', day_ix, ignore_vars=['Intercept', 'trial_ord'])
                    sig_tuned_ix_trl_tsk = get_perc_sig_model(hdf, m+'_trl_tsk', day_ix, ignore_vars=['Intercept', 'trial_ord'])
                    perc_tsk_mod_of_sig_cells_w_trl[dy].append(len(np.intersect1d(sig_tuned_ix_trl, sig_tuned_ix_trl_tsk))/float(len(sig_tuned_ix_trl)))
                    stat += 1

                    mod = getattr(hdf.root, m)
                    r2[dy].append(mod[day_ix]['r2'])
                    stat += 1
                    aic[dy].append(mod[day_ix]['aic'])
                    stat += 1
                    bic[dy].append(mod[day_ix]['bic'])
                    stat += 1
                except:
                    if stat < 5:
                        bic[dy].append(0)
                    if stat < 4:
                        aic[dy].append(0)
                    if stat < 3:
                        r2[dy].append(0)
                    if stat < 2:
                        perc_tsk_mod_of_sig_cells_w_trl[dy].append(0)
                    if stat < 1:
                        perc_tsk_mod_of_sig_cells[dy].append(0)


        #Plot dat.
        f2, ax = plt.subplots()
        ix = np.linspace(0, .6, 2*len(model_types))
        dx = ix[1]-ix[0]

        for i_d, dy in enumerate(n_day):
            Y = np.vstack((perc_tsk_mod_of_sig_cells[dy], perc_tsk_mod_of_sig_cells_w_trl[dy])).T.reshape(-1)

            for ib, (x, y) in enumerate(zip(ix, Y)):
                ax.bar(x+i_d, y, width=dx, color=cmap_list[ib/2], edgecolor=None)
        
        ax.set_xlabel(' Days, Different Models')
        ax.set_ylabel(' Percent Units a) Task-Modulated \nb) Task-Modulated after removing \ntrl_ord effect ')
        #Plot moar.
        f2, ax2 = plt.subplots()
        f2, ax3 = plt.subplots()
        f2, ax4 = plt.subplots()
        AX = [ax2, ax3, ax4]
        ix = np.linspace(0, .6, len(model_types))
        dx = ix[1]-ix[0]

        for ia, (axi, met, metnm) in enumerate(zip(AX, [r2, aic, bic], ['r2', 'aic', 'bic'])):
            for i_d, dy in enumerate(n_day):
                z = np.array([np.median(x) for x in met[dy]])
                axi.bar(ix+i_d, z, width=dx, color=cmap_list[i_d], edgecolor=None)
                axi.set_xlabel('Days')
                axi.set_ylabel(metnm)

def plot_model_results(filename, animal):
    dat = pickle.load(open(filename))
    f, ax = plt.subplots()
    f.set_figheight(10)
    f.set_figwidth(6)

    perc = dict(tun=[], tunpos=[], tuntemp=[])
    for i, j in enumerate(np.sort(dat.keys())):
        d = dat[j]
        tot = len(d['tun_no_task_mod'])+len(d['task_mod'])#+len(d['unt'])
        perc['tun'].append(len(d['task_mod'])/float(tot))
        perc['tunpos'].append(len(d['task_mod_wo_pos'])/float(tot))
        perc['tuntemp'].append(len(d['task_mod_wo_temp'])/float(tot))

    h1=ax.bar(0, np.mean(perc['tun']), width=.8, color=blue_cmap[0])
    ax.errorbar(0.4, np.mean(perc['tun']), np.std(perc['tun'])/np.sqrt(len(perc['tun'])), color=blue_cmap[0])

    h2 =ax.bar(1, np.mean(perc['tunpos']), width=.8, color=blue_cmap[1])
    ax.errorbar(1.4, np.mean(perc['tunpos']), np.std(perc['tunpos'])/np.sqrt(len(perc['tunpos'])), color=blue_cmap[1])

    h3 =ax.bar(2, np.mean(perc['tuntemp']), width=.8, color=blue_cmap[2])
    ax.errorbar(2.4, np.mean(perc['tuntemp']), np.std(perc['tuntemp'])/np.sqrt(len(perc['tuntemp'])), color=blue_cmap[2])

    ax.set_ylim([0., .8])
    ax.set_xlim([-.5, 3.5])
    ax.set_xticks([])
    ax.legend([h1, h2, h3], ['Task Mod. Velocity Tuning', 'Task Mod. Velocity Tuning\n After Pos. Removed', 'Task Mod. Velocity Tuning\n After Lead & Lag Removed'])
    ax.set_ylabel('Percent Tuned Units')
    plt.tight_layout()
    plt.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/'+animal+'_perc_tuned_units.pdf', bbox_inches='tight', pad_inches=1)

    f2, ax2 = plt.subplots()
    f2.set_figheight(10)
    f2.set_figwidth(8)
    md = dict(task_mod_wo_pos = [], task_mod = [], unt = [], tun_no_task_mod = [], task_mod_wo_temp=[])
    for i, j in enumerate(np.sort(dat.keys())):
        for L in ['tun_no_task_mod', 'task_mod', 'unt', 'task_mod_wo_pos', 'task_mod_wo_temp']:
            ix = np.array(d[L])
            if len(ix) > 0:
                arr = np.array(d['mod_depth'])
                md[L].append(arr[ix])

    h = []
    for ill, L in enumerate(['task_mod', 'task_mod_wo_pos', 'task_mod_wo_temp','tun_no_task_mod', 'unt']):
        if len(md[L]) > 0:
            md[L] = np.hstack((md[L]))
            h2 = ax2.bar(ill, np.mean(md[L]), color=grey_cmap[ill]/255.)
            h.append(h2)
            ax2.errorbar(ill+.4, np.mean(md[L]), np.std(md[L])/np.sqrt(len(md[L])), color=grey_cmap[ill]/255.)
    ax2.set_xticks([])
    ax2.legend(h, ['Task Mod.', 'Task Mod. Rm Pos.', 'Task Mod. Rm Leads & Lags', 'Tuned, No Task Mod.', 'Untuned'])
    ax2.set_ylabel('Modulation Depth')
    ax2.set_ylim([2.5, 5.5])
    plt.tight_layout()
    plt.savefig('/home/lab/preeya/fa_analysis/cosyne_figs/'+animal+'_tuned_units_vs_MD.pdf', bbox_inches='tight', pad_inches=1)

def plot_model_results_hdf_mega(filename = '~/grom_unit_metrics_hdf_mega.h5'):
    
    hdf = tables.openFile(filename)

    ##############################
    ######## PROCESSING ##########
    ##############################

    grp = list(hdf.root._f_walknodes('Group'))
    nms = []

    for t in range(1, len(grp)):
        tmp = grp[t]
        if 'nms' in tmp._v_pathname:
            nms.append(tmp._v_pathname[1:-4])
        else:
            print tmp._v_pathname

    # Initialize plots: 
    f, ax = plt.subplots()
    f2, ax2 = plt.subplots()
    f3, ax3 = plt.subplots()

    # Compute complexity of each point and future / history of it: 
    for i_d in range(9):
        n_params = []
        hist_fut = []
        res = []
        R2 = []
        R2_res = []
        AIC = []
        perc_task_mod = []
        perc_task_mod2 = []
        marker = []


        # v0_5 --> length 0, starting index = 5
        xv = re.compile('v\d_')
        xp = re.compile('p\d_')

        hlf = 2

        for im, nm in enumerate(nms):
            
            # Task or no? 
            if 'res' in nm: 
                res.append(1)
                r = True
            else:
                res.append(0)
                r = False

            #Number of parameters with vel: 
            m = xv.search(nm)
            vst = int(nm[m.end()]) # Starting index
            vnum = int(nm[m.start()+1])+1 # number of params

            m2 = xp.search(nm)

            #Number of parameters with pos:
            if m2 is not None:
                pst = int(nm[m2.end()]) # Starting index
                pnum = int(nm[m2.start()+1])+1 # number of params
                if pnum in [3, 4, 5]:
                    marker.append(1.)
                elif pnum in [0, 1, 2]:
                    marker.append(2.)
                else:
                    marker.append(3.)
            else:
                if vnum in range(5):
                    marker.append(-1.)
                else:
                    marker.append(0.)

            n_params.append(vnum)

            tab = getattr(hdf.root, nm)

            #day_ix = np.nonzero(tab[:]['day_ix']==i_d)[0]
            day_ix = np.arange(len(tab[:]['day_ix']))

            R2.append(np.nanmean(tab[day_ix]['r2'][:, 0]))
            AIC.append(np.nanmean(tab[day_ix]['aic'][:, 0]))

            if r:
                # Non - res: 
                tab2 = getattr(hdf.root, nm[4:])
                var = getattr(hdf.root, nm[4:]+'_nms')
                
                if np.logical_and('velx_tm0' in var.vars[:], 'vely_tm0' in var.vars[:]):
                    var_ix = np.array([i for i, j in enumerate(var.vars[:]) if j not in ['Intercept']])

                    #assert len(var_ix) == 2

                    pv = tab2[day_ix]['pvalue'][:, var_ix]
                    pv_ = np.zeros_like(pv)
                    pv_[pv <= 0.05] = 1
                    pv_[pv > 0.05] = 0
                    
                    neuron_ix = np.nonzero(np.sum(pv_, axis=1))[0]

                    var = getattr(hdf.root, nm+'_nms')
                    var_ix = np.array([i for i, j in enumerate(var.vars[:]) if j in ['velx_tm0:tsk', 'vely_tm0:tsk', 'tsk_vx', 'tsk_vy']])

                    ix_ = np.ix_(neuron_ix, var_ix)
                    pv = tab[day_ix]['pvalue'][ix_]
                    pv2_ = np.zeros_like(pv)
                    pv2_[pv > 0.05] = 0
                    pv2_[pv <= 0.05] = 1
                    neuron_ix_2 = np.nonzero(np.sum(pv2_, axis=1))[0]

                    if len(neuron_ix)==0:
                        perc_task_mod.append(0.)
                    else:
                        perc_task_mod.append(len(neuron_ix_2)/float(len(neuron_ix)))
                    perc_task_mod2.append(len(neuron_ix)/float(tab[:]['pvalue'].shape[0]))
                    R2_res.append(np.nanmean(tab[day_ix]['r2'][:, 0]))

            
                else:
                    perc_task_mod.append(-1.)
                    perc_task_mod2.append(-1.)
                    print 'unnecessary'
            else:
                perc_task_mod.append(-1.)
                perc_task_mod2.append(-2.)
                R2_res.append(-1.)

        #########################
        ######## PLOTS ##########
        #########################

        # Plots by increasing complexity
        # Color by History / Future / Neither

        order_n_param = np.argsort(n_params)
        res_ix_1 = np.nonzero(res)[0]
        res_ix_0 = np.nonzero(np.array(res)==0)[0]

        order_n_param_res_0 = np.argsort(np.array(n_params)[res_ix_0])
        order_n_param_res_1 = np.argsort(np.array(n_params)[res_ix_1])
        rr = np.array(R2)[res_ix_0][order_n_param_res_0]
        rr = rr/np.mean(rr)
        ax.plot(rr, color=cmap[i_d])
        ax.set_xlabel('Increasing Complexity of Model')
        ax.set_ylabel('R^2')

        rr_res = np.array(R2_res)[res_ix_1][order_n_param_res_1]
        rr_res = rr_res/np.mean(rr_res)
        ax3.plot(rr_res, color=cmap[i_d])

        ax2.plot(rr, rr_res, '.', color=cmap[i_d])
        slp, intc, rv, pv, ste = scipy.stats.linregress(rr, rr_res)
        xhat = np.linspace(.85, 1.05, 30)
        yhat = slp*xhat + intc
        ax2.plot(xhat, yhat, 'k-')
        ax2.set_title('PV: '+str(pv))

    ax.plot((np.array(marker)[res_ix_0][order_n_param_res_0]*.07) + 1., 'k*')

        # ### R2 -- for non-residuals
        # I = np.array(perc_task_mod)[res_ix_1][order_n_param_res_1]
        # I2 = np.array(perc_task_mod2)[res_ix_1][order_n_param_res_1]
        # #ax2.plot(I, '.', color=cmap[i_d])
        # ax2.plot(I2, '.', color=cmap[i_d])
        
        # ax2.set_xlabel('Increasing Complexity of Model')
        # ax2.set_ylabel('Percent Units Task Modulated')

def analyze_task_data(fname_tsk, fname_hdf):
    hdf = tables.openFile(fname_hdf)
    dat = pickle.load(open(fname))
    xv = re.compile('v\d_')
    xp = re.compile('p\d_')

    models = []
    score = []
    var_nms = []
    for m in dat.keys():
        if m[0] not in models:
            models.append(m[0])
            score.append(dat[(m[0], 0, 'score')])

            tab = getattr(hdf.root, m[0]+'_nms')
            var_nms.append(tab.vars[:])

    perc_correct = []
    perc_class_co = []
    plot_ix = []
    marker = []
    # Get percent accuracy:
    for m in models:

        j2 = xp.search(m)

        if j2 is not None:
            i = int(m[j2.start()+1])
            if i in [3, 4, 5]:
                marker.append(1.)
            elif i in [0, 1, 2]:
                marker.append(2.)
            else:
                print 'err'
        else:
            marker.append(-1.)

        X = dat[(m, 0, 'X')]
        prob = dat[(m, 0)]
        dprob = np.abs(np.diff(prob, axis=1))
        cutoff = np.percentile(dprob, 99)
        ix = np.nonzero(dprob>cutoff)[0]

        # metrics:
        plot_ix.append(ix)
        perc_correct.append(dat[(m, 0, 'score')])

    # Which models are most accurate? 
    for j, i in enumerate([-1., 1., 2]):
        ix = np.nonzero(np.array(marker) == i)[0]
        plt.plot(np.random.rand(len(ix)) + j, np.array(perc_correct)[ix], '.')

def compare_r2(hdf1, hdf2):
    grp = list(hdf1.root._f_walknodes('Group'))
    nms = []

    pvi = []
    pvi2 = []

    for t in range(1, len(grp)):
        tmp = grp[t]
        if 'nms' in tmp._v_pathname:
            nms.append(tmp._v_pathname[1:-4])
            tab = getattr(hdf1.root, nms[-1])
            pvi.append(tab[:]['param'][:, 0])
        else:
            print tmp._v_pathname

    grp = list(hdf2.root._f_walknodes('Group'))
    nms2 = []

    for t in range(1, len(grp)):
        tmp = grp[t]
        if 'nms' in tmp._v_pathname:
            nms2.append(tmp._v_pathname[1:-4])
            tab = getattr(hdf2.root, nms2[-1])
            pvi2.append(tab[:]['param'][:, 0])
        else:
            print tmp._v_pathname    

def get_individual_cell_tuning_curves_to_neural_push():
    tuning_dict = {}
    te_dict = {}
    day_dict = []
    tsk_dict = []
    d0 = dbfn.TaskEntry(4098)
    d0 = d0.date

    for i_d, day in enumerate(input_type):
        #snr = []
        for i_t, tsk in enumerate(day):
            for te in tsk:
                task_entry = dbfn.TaskEntry(te)
                te_dict[te] = task_entry
                if day[0][0] != 4558:
                    day_dict.append([te, (task_entry.date - d0).days])
                else:
                    day_dict.append([te, 44])
                tsk_dict.append([te, i_t])
    days = np.vstack((day_dict))
    tsks = np.vstack((tsk_dict))

    for i_d, day in enumerate(np.unique(days[:, 1])):
        ix = np.nonzero(days[:, 1]==day)[0]
        tes = days[ix, 0]   
        s = []
        v = [] 
        tsk = [] 
        n_units = te_dict[tes[0]].hdf.root.task[0]['spike_counts'].shape[0]

        for i_t, te in enumerate(tes):
            print te, 'Task: ', tsks[ix, 1][i_t]
            hdf = te_dict[te].hdf
            if hdf.root.task[0]['spike_counts'].shape[0] != n_units:
                print 'skipping ', te
            else:
                drives_neurons_ix0 = 3
                internal_state = hdf.root.task[:]['internal_decoder_state']
                update_bmi_ix = np.nonzero(np.diff(np.squeeze(internal_state[:, drives_neurons_ix0, 0])))[0]+1

                vel = hdf.root.task[update_bmi_ix-1]['internal_decoder_state'][:, [3, 5], 0]
                vel = hdf.root.task[update_bmi_ix]['internal_decoder_state'][:, [3, 5], 0]

                vel = np.hstack((vel, np.ones((len(vel), 1))))
                spks = hdf.root.task[:]['spike_counts'][:, :, 0]
                bspks = bin_spks(spks, update_bmi_ix)
                
                s.append(bspks)
                v.append(vel)
                tsk.append(np.zeros((len(bspks), )) + int(tsks[ix, 1][i_t]))

        s = np.vstack((s))
        v = np.vstack((v))
        tx = np.hstack((tsk))

        n_units = s.shape[1]
        tuning_dict[day] = {}

        tx0 = np.nonzero(tx==0)[0]
        tx1 = np.nonzero(tx==1)[0]

        for n in range(n_units):
            s0 = s[tx0, n] #Spikes: 
            s1 = s[tx1, n]

            x0 = np.linalg.lstsq(v[tx0, :], s0[:, np.newaxis]) #Regress Spikes against Velocities
            x1 = np.linalg.lstsq(v[tx1, :], s1[:, np.newaxis]) #Regress Spikes against Velocities
            
            q0 = s0[:, np.newaxis] - v[tx0, :]*np.mat(x0[0]) #Residuals
            q1 = s1[:, np.newaxis] - v[tx1, :]*np.mat(x1[0]) #Residuals
            
            #Explained Variance vs. Residual Variance: 
            ev0 = np.var(v[tx0, :]*np.mat(x0[0]))
            ev1 = np.var(v[tx1, :]*np.mat(x1[0]))

            tuning_dict[day][n] = [[ev0, np.var(q0), ev0/np.var(q0), x0[0]], [ev1, np.var(q1), ev1/np.var(q1), x1[0]]]
            #snr.append(ev/np.var(q))
        #tuning_dict[i_d]['snr'] = snr
    pickle.dump(tuning_dict, open(os.path.expandvars('$FA_GROM_DATA/co_obs_SNR_w_coefficients_by_task_corr.pkl'), 'wb'))

def plot_single_te(ax, day, task, pk, nfact=6):
    '''
    ax: axis to plot factors in generally a row of size 5, one for each factors
    day: int corresponding to day
    pk2: pickle dict w/ keys as [te, nfactors-1] --> FA model
    '''

    if task == 'co':
        task_ix = 0
    elif task == 'obs':
        task_ix = 1

    task_tes = input_type[day][task_ix]
    fa_keys = pk[day].keys()
    te_keys = []
    #Collect keys correspondng to greatest number of factors : 6 (6-1 = 5)
    for i, k in enumerate(fa_keys):
        #print 'int:', k[0], k[1]
        try:
            tmp = float(k[0])
            if int(k[0]) in task_tes:
                if int(k[1]) == nfact - 1:
                    te_keys.append(k)
        except:
            pass

    for ii, te in enumerate(te_keys):
        #Get FA model: 
        fa_tmp = pk[day][te]

        #Reduce FA model: 
        U = np.mat(fa_tmp.components_).T
        mu = pk[day]['mu', te[0]]
        A = U*U.T
        u, s, v = np.linalg.svd(A)
        ix = np.nonzero(np.cumsum(s)/float(np.sum(s))>.90)[0]
        print 'nf: ', ix[0]+1, te
        n_dim_main_shared = ix[0]+1
        trans_mat = u[:, :n_dim_main_shared].T
        trans_s = s[:n_dim_main_shared]
        # s_red = np.zeros_like(s)
        # s_hd = np.zeros_like(s)

        # s_red[:n_dim_main_shared] = s[:n_dim_main_shared]
        # main_shared_A = u*np.diag(s_red)*v
        # s_hd[n_dim_main_shared:] = s[n_dim_main_shared:]
  #       hd_shared_A = u*np.diag(s_hd)*v
        
  #       Psi = np.mat(np.zeros((U.shape[0], U.shape[0])))
  #       Psi[i] = fa_tmp.noise_variance_

  #       main_shared_B = np.linalg.inv(main_shared_A + hd_shared_A + Psi)
  #       main_sharL = main_shared_A*main_shared_B


        #Get te_num, spk_cnts, targ
        te_num = int(te[0])
        fname = os.path.expandvars('$FA_GROM_DATA/grom2016_co_obs_overlap_same_dec_diff_tasks_'+str(te_num)+'_spike_stuff.pkl')
        dat = pickle.load(open(fname))

        spks  = dict()
        for j in range(8):
            spks[j] = []

        targ_ix = dat['targ_ix']

        epoch_offset = int((te[0]-int(te[0]) -0.1)*100)*32
        if epoch_offset < 0:
            epoch_offset = 0
            epoch_num = np.arange(len(targ_ix))
        else:
            epoch_num = np.arange(32) + epoch_offset

        targ_ix_arr = np.array(targ_ix)

        for it, t in enumerate(targ_ix_arr[epoch_num]):
            #trans_ = fa_tmp.transform(dat['bin_spk'][it])
            tmp = dat['bin_spk'][it+epoch_offset].T

            #Subtract mean:
            zsc = tmp - np.tile(mu[0, :][:, np.newaxis], [1, tmp.shape[1]])

            trans_ = trans_mat*zsc
            spks[int(t)].append(trans_)

        #Now for each of these, plot modulations of first factor, second factor, etc: 
        for it, t in enumerate(np.unique(targ_ix)):
            D = spks[int(t)]
            max_ = 0
            for i, d in enumerate(D):
                max_ = np.max([d.shape[1], max_])

            S = np.zeros((n_dim_main_shared, len(D), max_))
            for j, d in enumerate(D):
                S[:, j, :d.shape[1]] = d
                S[:, j, d.shape[1]:] = np.nan

            for j in range(n_dim_main_shared):
                d = np.nanmean(S[j, :, :], axis=1)
                sm = np.nanstd(S[j, :, :], axis=1)/np.sqrt(S.shape[1])
                x = np.arange(len(d))
                ax[j].plot(t, np.mean(d), '.', markersize=20, color=cmap_list[ii])
                ax[j].set_ylim([-5, 5])
                #ax[i].fill_between(x, d-sm, d+sm, color=cmap_list[it], alpha=0.5)
                #ax[i].plot(x, d, color=cmap_list[it], linewidth=.5)

#### NEW PLOT FCNS ######


def plot_sweep_alpha(animal, alphas = None, model_set_number = 1, ndays=None, skip_plots = True):

    if model_set_number == 1:
        # model_names = ['prespos_0psh_1spksm_0_spksp_0', 'prespos_1psh_1spksm_0_spksp_0', 'hist_fut_4pos_1psh_1spksm_0_spksp_0',
        # 'hist_fut_4pos_1psh_1spksm_1_spksp_0','hist_fut_4pos_1psh_1spksm_1_spksp_1']
        model_names = ['prespos_0psh_1spksm_0_spksp_0', 
                            'hist_1pos_-1psh_1spksm_0_spksp_0', 
                            'hist_1pos_1psh_1spksm_0_spksp_0',
                            'hist_1pos_2psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_0_spksp_0', 
                            'hist_1pos_3psh_1spksm_1_spksp_0']
    elif model_set_number in [2, 3]:
        # model_names = ['hist_1pos_0psh_1spksm_0_spksp_0', 'hist_1pos_0psh_1spksm_1_spksp_0', 'hist_4pos_0psh_1spksm_0_spksp_0', 'hist_4pos_0psh_1spksm_1_spksp_0',
        # 'hist_4pos_0psh_1spksm_4_spksp_0', 'prespos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_0spksm_1_spksp_0']
        model_names = ['prespos_0psh_0spksm_1_spksp_0','hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_1spksm_0_spksp_0']

    elif model_set_number in [4]:
        model_names = ['prespos_0psh_1spksm_0_spksp_0', 
                       'prespos_0psh_1spksm_1_spksp_0',

                       'prespos_1psh_1spksm_0_spksp_0', 
                       'prespos_1psh_1spksm_1_spksp_0',
                       
                       'hist_1pos_1psh_1spksm_0_spksp_0', 
                       'hist_1pos_1psh_1spksm_1_spksp_0', 
                       
                       'hist_2pos_1psh_1spksm_0_spksp_0', 
                       'hist_2pos_1psh_1spksm_1_spksp_0']

    elif model_set_number in [5]:
        model_names = ['hist_1pos_1psh_0spks_0_spksp_0',
                       'hist_1pos_0psh_1spks_1_spksp_0',
                       'hist_1pos_0psh_0spks_1_spksp_0',
                       'hist_1pos_1psh_0spks_1_spksp_0',
                       'hist_1pos_1psh_1spks_1_spksp_0']

    elif model_set_number in [6]:
        model_names = ['hist_1pos_1psh_0spks_0_spksp_0'] ### Only state; 

    elif model_set_number == 7:
        model_names = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_0psh_1spksm_1_spksp_0']

    elif model_set_number == 8:
        model_names = ['hist_1pos_0psh_0spksm_1_spksp_0', 'hist_1pos_1psh_0spksm_0_spksp_0', 'hist_1pos_3psh_0spksm_0_spksp_0']

    if animal == 'grom':
        hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_sweep_alpha_days_models_set%d.h5' %model_set_number)
        if ndays is None:
            ndays = 9; 
    
    elif animal == 'jeev':
        hdf = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_sweep_alpha_days_models_set%d.h5' %model_set_number)
        if ndays is None:
            ndays = 4; 

    if alphas is None:
        alphas = np.hstack(([[10**float(i), 5*10**float(i)] for i in np.arange(-3, 7)]))
    
    str_alphas = [];
    for ia, alpha in enumerate(alphas):
        tmp = str(alpha)
        tmp = tmp.replace('.','_')
        str_alphas.append(tmp)

    max_alpha = dict()

    #### days #####
    for i_d in range(ndays):

        #### model names ####
        for im, model_nm in enumerate(model_names):

            if skip_plots:
                pass
            else:
                f, ax = plt.subplots(figsize = (5, 3)) ### Separate fig for each day/
            max_alpha[i_d, model_nm] = []

            alp = []; 
            #### alphas ####
            for ia, (alpha, alpha_float) in enumerate(zip(str_alphas, alphas)):
                alp_i = []; 

                for fold in range(5):
                    tbl = getattr(hdf.root, model_nm+'_alpha_'+alpha+'_fold_'+str(fold))
                    day_index = np.nonzero(tbl[:]['day_ix'] == i_d)[0]

                    #### Add day index to alphas
                    alp_i.append(tbl[day_index]['r2'])

                ### plot alpha vs. mean day: 
                if skip_plots:
                    pass
                else:
                    ax.bar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), width=.3)
                    ax.errorbar(np.log10(alpha_float), np.nanmean(np.hstack((alp_i))), np.nanstd(np.hstack((alp_i)))/np.sqrt(len(np.hstack((alp_i)))),
                        marker='|', color='k')

                ### max_alpha[i_d, model_nm] = []
                alp.append([alpha_float, np.nanmean(np.hstack((alp_i)))])

            if skip_plots:
                pass
            else:
                ax.set_title('Animal %s, Model %s, Day %d' %(animal, model_nm, i_d), fontsize = 10)
                ax.set_ylabel('R2')
                ax.set_xlabel('Log10(alpha)')
                f.tight_layout()

            ### Get the max to use: 
            alp = np.vstack((alp))
            ix_max = np.argmax(alp[:, 1])
            max_alpha[i_d, model_nm] = alp[ix_max, 0]

    return max_alpha

def plot_new_linear_tuning():
    
    #hdf = tables.openFile('/Volumes/TimeMachineBackups/grom2016/grom2017_new_linear_tuning_subset.h5')
    #if animal == 'grom':
    try:
        pre = '/Volumes/TimeMachineBackups'
        hdf_g = tables.openFile(pre+'/grom2016/grom2018_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5')
        #elif animal == 'jeev':
        hdf_j = tables.openFile(pre+'/jeev2013/jeev2018_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5')
    
    except:
        pre = '/Users/preeyakhanna/Dropbox/TimeMachineBackups'
        
        #hdf_g = tables.openFile(pre+'/grom2016/grom2017_new_linear_tuning_subset_min4_to_pos4.h5')
        hdf_g = tables.openFile(pre+'/grom2016/grom2018_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5')
        
        #elif animal == 'jeev':
        #hdf_j = tables.openFile(pre+'/jeev2013/jeev2017_test_new_linear_tuning_subset_min4_to_pos4.h5')
        hdf_j = tables.openFile(pre+'/jeev2013/jeev2017_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5')

    names = ['hist_1', 'hist_2', 'hist_3', 'hist_4', 'pres', 'fut_1', 'fut_2', 'fut_3', 'fut_4',
        'hist_fut_1', 'hist_fut_2', 'hist_fut_3', 'hist_fut_4']

    fig_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

    # R2 model vs. R2 task
    colors = ['k', 'k']
    # for ih, (hdf, hdf_nm) in enumerate(zip([hdf_g, hdf_j], ['MonkeyG', 'MonkeyJ'])):
    #     f, ax = plt.subplots()
    #     X = []
    #     Y = []
        
    #     for i_d in range(int(np.max(hdf.root.fut_2pos_0[:]['day_ix'])+1)):
    #         ix = np.nonzero(hdf.root.fut_2pos_0[:]['day_ix']==i_d)[0]
    #         x = []
    #         y = []
    #         for pos in range(2):
    #             for n, nm in enumerate(names):
    #                 nm_pos = nm + 'pos_' + str(pos)
    #                 tbl = getattr(hdf.root, nm_pos)
    #                 r2_mn = np.nanmean(tbl[ix]['r2'])
                
    #                 tbl_res = getattr(hdf.root, 'res_'+nm_pos)
    #                 r2_mn_res = np.nanmean(tbl_res[ix]['r2'])

    #                 x.append(r2_mn)
    #                 y.append(r2_mn_res)
    #         X.append(np.hstack((x))/np.mean(np.hstack((x))))
    #         Y.append(np.hstack((y))/np.mean(np.hstack((y))))
    #     ax.plot(np.hstack((X)), np.hstack((Y)), '.', color=colors[ih])
    #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.hstack((X)), np.hstack((Y)))
    #     print hdf_nm, p_value,  'n = ', len(np.hstack((X)))
    #     xhat = np.linspace(0.7, 1.4, 100)
    #     yhat = intercept + slope*xhat
    #     ax.plot(xhat, yhat, '-', color=colors[ih])
    #     if ih == 0:
    #         ax.set_xlim([.75, 1.2])
    #         ax.set_ylim([.85, 1.15])
    #     if ih == 1:
    #         ax.set_xlim([.7, 1.4])
    #         ax.set_ylim([.7, 1.2])

    #     ax.set_xlabel('Normalized R^2, Model')
    #     ax.set_ylabel('Normalized R^2, Model Residuals w/ Task Factor')
    #     ax.set_title(hdf_nm+' pvalue:'+str(p_value))
    #     #f.savefig(fig_dir+hdf_nm+'r2_model_vs_r2_res_tsk_factor.svg', transparent=True)
    
    ##############################
    ##### FIGURE 4F ##############
    ##############################
    #import seaborn
    colors = ['k', 'k']
    ylims = dict()
    ylims['Monkey G'] = [0.0024, 0.0026]
    ylims['Monkey J'] = [0.0, 0.02]
    NN = dict()
    NN['Monkey G'] = 619.
    NN['Monkey J'] = 81.

    symbols = ['0','-','+','-','+','-','+','-','+','-/+','-/+','-/+','-/+']

    import matplotlib
    cmap = matplotlib.cm.get_cmap('Purples')

    for ih, (hdf, hdf_nm) in enumerate(zip([hdf_g, hdf_j], ['Monkey G', 'Monkey J'])):
        nlags = []
        f2, ax2 = plt.subplots()
        f2.set_figheight(5)
        f2.set_figwidth(6)
        tmp = {}
        tmp[0] = []
        tmp[1] = []

        nsig = {}
        nsig[0] = []
        nsig[1] = []

        N = float(len(hdf.root.fut_2pos_0))
        for pos in range(1, 2):
            tbl = getattr(hdf.root, 'res_prespos_'+str(pos))
            tmp[pos].append(np.nanmean(tbl[:]['r2']))

            # How many of the neurons have % significant? 
            nms = getattr(hdf.root, 'res_prespos_'+str(pos)+'_nms')
            ix = [j for j, i in enumerate(nms.vars[:]) if 'tsk' in i]

            # Number of significant neurons: 
            pv_task = hdf.root.res_prespos_0[:]['pvalue'][:, ix]
            sig = np.any(pv_task<0.05, axis=1)
            nsig[pos].append(sig.sum())

        nlags.append(0)

        offs = 1

        # Get only-future or only-history:
        # How many params
        for nparm in range(1, 5):

            # History or future
            for ic, caus in enumerate(['hist', 'fut']):

                # Has position or not
                for ip, pos in enumerate(range(1, 2)):

                    nm_pos = 'res_'+caus+'_' +str(nparm)+'pos_' + str(pos)
                    tbl = getattr(hdf.root, nm_pos)
                    tmp[pos].append(np.nanmean(tbl[:]['r2']))
                    
                    # How many of the neurons have % significant? 
                    nms = getattr(hdf.root, nm_pos+'_nms')
                    ix = [j for j, i in enumerate(nms.vars[:]) if 'tsk' in i]

                    # Number of significant neurons: 
                    tbl_task = getattr(hdf.root, nm_pos)
                    pv_task = tbl_task[:]['pvalue'][:, ix]
                    sig = np.any(pv_task<0.05, axis=1)
                    nsig[pos].append(sig.sum())

                nlags.append(nparm)

                offs += 1

        # Get combo future history:
        for nparm in range(1, 5):
            for pos in range(1, 2):

                nm_pos = 'res_hist_fut_' + str(nparm)+'pos_' + str(pos)
                tbl = getattr(hdf.root, nm_pos)
                tmp[pos].append(np.nanmean(tbl[:]['r2']))

                # How many of the neurons have % significant? 
                nms = getattr(hdf.root, nm_pos+'_nms')
                ix = [j for j, i in enumerate(nms.vars[:]) if 'tsk' in i]

                # Number of significant neurons: 
                tbl_task = getattr(hdf.root, nm_pos)
                pv_task = tbl_task[:]['pvalue'][:, ix]
                sig = np.any(pv_task<0.05, axis=1)
                nsig[pos].append(sig.sum())

            nlags.append(nparm*2)
            offs += 1

        #x = ax2.plot(tmp[0], '.--', color=colors[ih])
        xp = ax2.plot(tmp[1], '.-', color=colors[ih])

        # Fill in colors:
        import matplotlib.patches as mpatches
        for t in range(len(tmp[1])):
            dylim = ylims[hdf_nm][1] - ylims[hdf_nm][0]
            rect = mpatches.Rectangle((t-0.5, ylims[hdf_nm][0]), 1., dylim, edgecolor='gray',
                facecolor=cmap(nlags[t]/8.), alpha=0.5)
            ax2.add_patch(rect)
            
            #ax2.text(t, ylims[hdf_nm][1]-(.1*dylim), symbols[t], fontsize=8, horizontalalignment='center')
            # ax2.text(t, ylims[hdf_nm][1]-(.1*dylim), str(np.round(100*float(nsig[0][t])/float(NN[hdf_nm]))), fontsize=8, 
            #     horizontalalignment='center')
            ax2.text(t, ylims[hdf_nm][1]-(.1*dylim), str(int(np.round(100*float(nsig[1][t])/float(NN[hdf_nm]))))+'%', fontsize=8, 
                horizontalalignment='center')

        ax2.set_ylim(ylims[hdf_nm])
        ax2.set_xlim([-0.5, t+.5])
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.set_ylabel('R2')
        ax2.set_title(hdf_nm)
        f2.savefig(fig_dir+hdf_nm+'_residual_pos_and_vel_only_R2_neur_true.svg', transparent=True)



    ##############################
    ##### FIGURE 4E ##############
    ##############################
    #import seaborn
    colors = ['k', 'k']
    ylims = dict()
    ylims['Monkey G'] = [0.05, 0.08]
    ylims['Monkey J'] = [0.12, 0.18]

    symbols = ['0','-','+','-','+','-','+','-','+','-/+','-/+','-/+','-/+']

    import matplotlib
    cmap = matplotlib.cm.get_cmap('Purples')

    for ih, (hdf, hdf_nm) in enumerate(zip([hdf_g, hdf_j], ['Monkey G', 'Monkey J'])):
        nlags = []
        f2, ax2 = plt.subplots()
        f2.set_figheight(5)
        f2.set_figwidth(6)
        tmp = {}
        tmp[0] = []
        tmp[1] = []

        N = float(len(hdf.root.fut_2pos_0))
        for pos in range(2):
            tbl = getattr(hdf.root, 'prespos_'+str(pos))
            tmp[pos].append(np.nanmean(tbl[:]['r2']))

        nlags.append(0)

        offs = 1

        # Get only-future or only-history:
        # How many params
        for nparm in range(1, 5):

            # History or future
            for ic, caus in enumerate(['hist', 'fut']):

                # Has position or not
                for ip, pos in enumerate(range(2)):

                    nm_pos = caus+'_' +str(nparm)+'pos_' + str(pos)
                    tbl = getattr(hdf.root, nm_pos)
                    tmp[pos].append(np.nanmean(tbl[:]['r2']))

                nlags.append(nparm)

                offs += 1

        # Get combo future history:
        for nparm in range(1, 5):
            for pos in range(2):

                nm_pos = 'hist_fut_' + str(nparm)+'pos_' + str(pos)
                tbl = getattr(hdf.root, nm_pos)
                tmp[pos].append(np.nanmean(tbl[:]['r2']))
            nlags.append(nparm*2)
            offs += 1

        x = ax2.plot(tmp[0], '.--', color=colors[ih])
        xp = ax2.plot(tmp[1], '.-', color=colors[ih])

        # Fill in colors:
        import matplotlib.patches as mpatches
        for t in range(len(tmp[0])):
            dylim = ylims[hdf_nm][1] - ylims[hdf_nm][0]
            rect = mpatches.Rectangle((t-0.5, ylims[hdf_nm][0]), 1., dylim, edgecolor='gray',
                facecolor=cmap(nlags[t]/8.), alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(t, ylims[hdf_nm][1]-(.1*dylim), symbols[t], fontsize=8, horizontalalignment='center')
        ax2.set_ylim(ylims[hdf_nm])
        ax2.set_xlim([-0.5, t+.5])
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.set_ylabel('R2')
        f2.savefig(fig_dir+hdf_nm+'_R2_linear_tuning_pos_and_vel_norm_neur_true.svg', transparent=True)
     
def plot_new_linear_tuning_w_action_neural(plot_days = True):

    ### Ridge regression ###

    ########################
    ##### GROM, day 0 ######
    ########################
    ### Neural y_{t-1} AND y_{t+1} for diff state lags w/ NP_t vs. no neural for diff state lags w/ NP_t; 
    # hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_13_19_21_34.h5')
    # title = 'Est. y_t | y_{t-1}, y_{t+1}, np_{t}, various states'

    # ### Neural y_{t-1} for diff state lags w/ NP_t vs. no neural for diff state lags w/ NP_t; 
    # hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_13_19_23_13.h5')
    # title = 'Est. y_t | y_{t-1}, np_{t}, various states'

    # ### Neural y_{t} for diff state lags w/ NP_t vs. no neural for diff state lags w/ NP_t:
    # ### Sanity check to make sure this is 100% --> predicting neural data given neural data; 
    # hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_13_19_25_11.h5')
    # title = 'Est.y_t | y_t...'

    ########################
    ##### GROM, ALL ######## -- all days used alpha == 1.0
    ### Neural y_{t-1} AND y_{t+1} for diff state lags w/ NP_t vs. no neural for diff state lags w/ NP_t; 
    ########################
    # hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_13_19_58_00.h5')
    # title = 'Est. y_t | y_{t-1}, y_{t+1}, np_{t}, various states'
    # hdf_j = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_2019_09_13_19_37_36.h5')
    
    # #### Specialized alphas used for each day: 
    hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_15_16_30_10.h5')
    hdf_j = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_2019_09_15_16_47_04.h5')
    title = 'Est. y_t | y_{t-1}, y_{t+1}, np_{t}, various states'


    #### Specialized alphas used for each day AND neural push lags used as well: 
    hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_16_07_36_41.h5')
    hdf_j = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_2019_09_16_07_32_20.h5')
    title = 'Est. y_t | y_{t-1}, y_{t+1}, np_{t} W LAGS, various states'

    #### Specialized alphas used for each day AND neural push lags used AND ONLY y_{t-1} used; 
    hdf_g = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom_2019_09_16_08_28_05.h5')
    hdf_j = tables.openFile('/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev_2019_09_16_08_32_19.h5')
    title = 'Est. y_t | y_{t-1}, np_{t-T},...np{t+T}, s_{t-T},...s{t+T}'



    names = ['hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5', 'pres', 'fut_1', 'fut_2', 'fut_3', 'fut_4', 'fut_5',
        'hist_fut_1', 'hist_fut_2', 'hist_fut_3', 'hist_fut_4', 'hist_fut_5']
    fig_dir = '/Users/preeyakhanna/Dropbox/Carmena_Lab/Documentation/BMI_co_obs_paper/figures/data_figs/'

    ##############################
    ##### FIGURE 4F ##############
    ##############################
    #import seaborn
    colors = ['k', 'b', 'r']
    marks = ['.-', '.--']

    ylims = dict()
    ylims['Monkey G'] = [0., 0.15]
    ylims['Monkey J'] = [0.1, 0.25]
    NN = dict()
    NN['Monkey G'] = 619.
    NN['Monkey J'] = 81.

    ndays = dict();
    ndays['Monkey G'] = 9;
    ndays['Monkey J'] = 4; 

    symbols = ['0','-','+','-/+', '-','+','-/+', '-','+','-/+', '-','+','-/+']

    import matplotlib
    cmap = matplotlib.cm.get_cmap('Purples')

    for ih, (hdf, hdf_nm) in enumerate(zip([hdf_g, hdf_j], ['Monkey G', 'Monkey J'])):
        nlags = []
        f2, ax2 = plt.subplots()
        f2.set_figheight(5)
        f2.set_figwidth(6)
        tmp = {}
        for psh in range(2):
            for pos in range(2):
                for neur in range(2):
                    if plot_days:
                        for day in range(ndays[hdf_nm]):    
                            tmp[psh, pos, neur, day] = []
                    else:
                        tmp[psh, pos, neur, 0] = []

        N = float(len(hdf.root.fut_2pos_0psh_0spks_1_fold_0))

        ### Add the present: 
        for psh in range(1, 2):
            for ip, pos in enumerate(range(1, 2)):
                for neur in range(2):
                    tmpi = dict()
                    for fold in range(5):
                        tbl = getattr(hdf.root, 'prespos_'+str(pos)+'psh_'+str(psh)+'spks_'+str(neur)+'_fold_'+str(fold))
                        
                        if plot_days:
                            dyz = np.unique(tbl[:]['day_ix'])
                            for day_i in np.unique(tbl[:]['day_ix']):
                                ix = np.nonzero(tbl[:]['day_ix'] == day_i)[0]
                                if day_i in tmpi.keys():
                                    tmpi[day_i].append(tbl[ix]['r2'])
                                else:
                                    tmpi[day_i] = [tbl[ix]['r2']]
                        else:
                            dyz = [0]
                            try:
                                tmpi[0].append(tbl[:]['r2'])
                            except:
                                tmpi[0] = [tbl[:]['r2']]


                    for day_i in dyz:
                        tmpi[day_i] = np.nanmean(np.nanmean(np.vstack((tmpi[day_i])), axis=0))

                        if np.any(np.isinf(tmpi[day_i])):
                            import pdb; pdb.set_trace()

                        tmp[psh, pos, neur, day_i].append(tmpi[day_i])
                    
                    nlags.append(0)

                    # Get only-future or only-history:
                    # How many params
                    for nparm in range(1, 5):

                        # History or future
                        for ic, caus in enumerate(['hist', 'fut']):

                            tmpi = dict(); 
                            for fold in range(5):
                                # Has position or not
                                nm_pos = caus+'_' +str(nparm)+'pos_' + str(pos)+'psh_'+str(psh)+'spks_'+str(neur)+'_fold_'+str(fold)
                                tbl = getattr(hdf.root, nm_pos)

                                if plot_days: 
                                    for day_i in np.unique(tbl[:]['day_ix']):
                                        ix = np.nonzero(tbl[:]['day_ix'] == day_i)[0]
                                        if day_i in tmpi.keys():
                                            tmpi[day_i].append(tbl[ix]['r2'])
                                        else:
                                            tmpi[day_i] = [tbl[ix]['r2']]
                                else:
                                    try:
                                        tmpi[0].append(tbl[:]['r2'])
                                    except:
                                        tmpi[0] = [tbl[:]['r2']]
                            
                            for day_i in dyz:
                                tmpi[day_i] = np.nanmean(np.nanmean(np.vstack((tmpi[day_i])), axis=0))
                                tmp[psh, pos, neur, day_i].append(np.nanmean(tmpi[day_i]))

                                if np.any(np.isinf(tmp[psh, pos, neur, day_i])):
                                    import pdb; pdb.set_trace()
                            nlags.append(nparm)

                        # Get combo future history:
                        tmpi = dict() 
                        for fold in range(5):
                            nm_pos = 'hist_fut_' + str(nparm)+'pos_' + str(pos)+'psh_'+str(psh)+'spks_'+str(neur)+'_fold_'+str(fold)
                            tbl = getattr(hdf.root, nm_pos)

                            if plot_days:
                                for day_i in np.unique(tbl[:]['day_ix']):
                                    ix = np.nonzero(tbl[:]['day_ix'] == day_i)[0]
                                    if day_i in tmpi.keys():
                                        tmpi[day_i].append(tbl[ix]['r2'])
                                    else:
                                        tmpi[day_i] = [tbl[ix]['r2']]
                            else:
                                try:
                                    tmpi[0].append(tbl[:]['r2'])
                                except:
                                    tmpi[0] = [tbl[:]['r2']]

                        for day_i in dyz:
                            tmpi[day_i] = np.nanmean(np.nanmean(np.vstack((tmpi[day_i])), axis=0))
                            tmp[psh, pos, neur, day_i].append(np.nanmean(tmpi[day_i]))

                            if np.any(np.isinf(tmp[psh, pos, neur, day_i])):
                                import pdb; pdb.set_trace()
                        nlags.append(nparm*2)
                    
                    for day_i in dyz:
                        x = ax2.plot(tmp[psh, pos, neur, day_i], marks[neur], color=cmap_list[int(day_i)])

        # Fill in colors:
        import matplotlib.patches as mpatches
        for t in range(len(tmp[1, 1, 0, 0])):
            dylim = ylims[hdf_nm][1] - ylims[hdf_nm][0]
            rect = mpatches.Rectangle((t-0.5, ylims[hdf_nm][0]), 1., dylim, edgecolor='gray',
                facecolor=cmap(nlags[t]/8.), alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(t, ylims[hdf_nm][1]-(.1*dylim), symbols[t], fontsize=8, horizontalalignment='center')
        ax2.set_ylim(ylims[hdf_nm])
        #ax2.set_xlim([-0.5, t+.5])
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.set_ylabel('R2')
        ax2.set_title(title,fontsize=12)
        f2.tight_layout()
        
def test():
    import co_obs_tuning_matrices
    from resim_ppf import file_key 
    grom_input_type = analysis_config.data_params['grom_input_type']
    jeev_input_type = file_key.task_filelist

    # Main fcns:
    plot_factor_tuning_curves.model_individual_cell_tuning_curves(jeev_input_type, '/Users/preeyakhanna/Dropbox/TimeMachineBackups/jeev2013/jeev2017_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5', 
        history_bins_max=4, only_vx0_vy0_tsk_mod=True, animal='jeev',**dict(ridge=False, normalize_neurons=True))

    plot_factor_tuning_curves.model_individual_cell_tuning_curves(grom_input_type, '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/grom2018_OLS_linear_tuning_min4_to_pos4_vel_is_cursor_state_norm_neur_True.h5', 
        history_bins_max=4, only_vx0_vy0_tsk_mod=True, animal='grom',**dict(ridge=False, normalize_neurons=True))

def ang_difference(a1, a2):
    """Compute the smallest difference between two angle arrays.
    Parameters
    ----------
    a1, a2 : np.ndarray
        The angle arrays to subtract
    deg : bool (default=False)
        Whether to compute the difference in degrees or radians
    Returns
    -------
    out : np.ndarray
        The difference between a1 and a2
    """

    diff = a1 - a2
    return wrapdiff(diff)

def wrapdiff(diff, deg=False):
    """Given an array of angle differences, make sure that they lie
    between -pi and pi.
    Parameters
    ----------
    diff : np.ndarray
        The angle difference array
    deg : bool (default=False)
        Whether the angles are in degrees or radians
    Returns
    -------
    out : np.ndarray
        The updated angle differences
    """

    base = np.pi * 2
    i = np.abs(diff) > (base / 2.0)
    out = diff.copy()
    out[i] -= np.sign(diff[i]) * base
    return out

### What do the actual task differences look like: 
def plot_a_t_given_a_tm1(min_obs = 10):

    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    for ia, (animal, input_type) in enumerate(zip(['grom'], [analysis_config.data_params['grom_input_type']])):#, 'jeev'], [analysis_config.data_params['grom_input_type'], file_key.task_filelist])):

        for i_d, day in enumerate([input_type[0]]):

            f, ax = plt.subplots(ncols = 8, nrows = 4, figsize=(16, 8))
            command_bin_dist = []; 

            tp1_dict = dict(); t0_dict = dict(); 
            for tsk in range(2):
                for targ in range(8):
                    for m in range(4):
                        for a in range(8):
                            tp1_dict[tsk, targ, m, a] = []
                            t0_dict[tsk, targ, m, a] = []

            for i_t, tsk in enumerate(day):
                commands = []; 
                targets = []; 
                trials = []; 
                trl_cnt = 0; 

                ### gather all of these and put on a plot: 
                for te_num in tsk: 
                    bin_spk_nonz, targ_ix, trial_ix_all, KG, decoder_all, cursor_state = fit_LDS.pull_data(te_num, animal, pre_go=0, 
                        binsize_ms=100, keep_units='all')

                    commands.extend(decoder_all)
                    targets.extend(targ_ix)
                    trials.extend(trial_ix_all + trl_cnt)

                    trl_cnt += np.max(trial_ix_all) + 1; 

                ### Convert to commands via subspace overlap
                commands_disc = subspace_overlap.commands2bins(commands, mag_boundaries, animal, i_d, vel_ix = [3, 5])
                targets = np.hstack((targets))
                trials = np.hstack((trials))

                ### Go through each command and figure out its bin, then add the next REAL entry in 
                for i_trl, trl in enumerate(commands_disc):
                    for i_bin, (binm, bina) in enumerate(trl[:-1, :]): 
                        ### Which target are you in? 
                        trial_ix = np.nonzero(trials == i_trl)[0]
                        targ = targets[trial_ix[3]]

                        ### Add the next trial
                        tp1_dict[i_t, targ, binm, bina].append( np.squeeze(np.array(commands[i_trl][i_bin+1, [3, 5]] )))
                        t0_dict[i_t, targ, binm, bina].append(  np.squeeze(np.array(commands[i_trl][i_bin, [3, 5]]   )))

            for m in range(4):
                for a in range(8):

                    ### Plot the typical 
                    mn_m = [0] + mag_boundaries[animal, i_d] + [3]
                    mn_m = np.mean([mn_m[m+1], mn_m[m]])
                    A = np.linspace(0., 2*np.pi, 9) - np.pi/8
                    mn_a = np.mean(A[[a, a+1]])
                    ax[m, a].plot([0, mn_m*np.cos(mn_a)], [0, mn_m*np.sin(mn_a)], '-', color='k', linewidth=.5)

                    for tsk, tsk_col in enumerate(['b', 'r']):

                        tsk_mn = []; 

                        for _, (targ, targ_alph) in enumerate(zip(np.arange(8), np.linspace(1., 1., 8.))):

                            if len(tp1_dict[tsk,targ,m,a]) > 0:
                                tp1 = np.vstack(( tp1_dict[tsk, targ, m, a] ))

                                if tp1.shape[0] >= min_obs:
                                    tp1 = np.mean(tp1, axis=0)
                                    ax[m, a].plot([0, tp1[0]], [0, tp1[1]], '-', color=tsk_col, alpha=targ_alph, linewidth=.5)
                                else:
                                    print 'skipping N = %d' %(tp1.shape[0])
                                ax[m, a].set_xlim([-3, 3])
                                ax[m, a].set_ylim([-3, 3])
                                
                                tsk_mn.extend(tp1_dict[tsk, targ, m, a])
                                command_bin_dist.append(np.vstack(( tp1_dict[tsk, targ, m, a] )).shape[0])
                            
                        tsk_mn = np.mean(np.vstack((tsk_mn)), axis=0)
                        #ax[m, a].plot([0, tsk_mn[0]], [0, tsk_mn[1]], '-', color=tsk_col, alpha=.5, linewidth = 3.)

                    ax[m, a].set_title('M %d, A %d' %(m, a))
            f.tight_layout()
            import pdb; pdb.set_trace()
            f, ax = plt.subplots()
            binz = np.arange(100)
            h, i = np.histogram(np.hstack((command_bin_dist)), binz)
            ax.plot(i[:-1] + .5, h)

### Birthday plots
def plot_y_t_for_conds():
    model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_grom_model_set1.pkl', 'rb'))
    
    i_d = 0; animal = 'grom' 
    neural_push = model_dict[(i_d, 'np')]

    ### Want to understand what causes this model to look bad when avg. over task;
    pred_spiking0 = model_dict[(i_d, 'hist_fut_4pos_1psh_1spksm_0_spksp_0')]
    pred_spiking1 = model_dict[(i_d, 'hist_fut_4pos_1psh_1spksm_1_spksp_0')]

    spks = model_dict[(i_d, 'spks')]
    ### Mag
    mag_boundaries = pickle.load(open('/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/radial_boundaries_fit_based_on_perc_feb_2019.pkl'))

    ### Convert to commands via subspace overlap
    commands_disc = subspace_overlap.commands2bins([model_dict[i_d, 'np']], mag_boundaries, animal, i_d, vel_ix = [0, 1])[0]
    tsk = model_dict[i_d, 'task']
    trg = model_dict[i_d, 'trg']

    N = model_dict[i_d, 'spks'].shape[1]

    marker = ['.', 'd']
    color = ['maroon', 'orangered', 'goldenrod','olivedrab','teal', 'steelblue', 'midnightblue', 'darkmagenta',]

    for n in range(35, N):

        f, ax = plt.subplots(ncols = 8, nrows = 4, figsize = (14, 7))
        
        for i_mag in range(4):
            for i_ang in range(8):
                axi = ax[i_mag, i_ang]

                for task in range(2):
                    T = []; P = []; ymin = 0; ymax = 0; 
                
                    for targ in range(8):
                        ix = (commands_disc[:, 0] == i_mag) & (commands_disc[:, 1] == i_ang) & (tsk == task) & (trg == targ) 
                        ix = np.nonzero( ix == True)[0]
                
                        if len(ix) > 5:
                            axi.plot(np.mean(spks[ix, n]), np.mean(pred_spiking0[ix, n]), marker=marker[task], color=color[targ], alpha=0.3)
                            axi.plot(np.mean(spks[ix, n]), np.mean(pred_spiking1[ix, n]), marker=marker[task], color=color[targ], alpha=1.0)
                            axi.plot([np.mean(spks[ix, n]), np.mean(spks[ix, n])], [np.mean(pred_spiking0[ix, n]), np.mean(pred_spiking1[ix, n])],'-',
                                color=color[targ])
                            ymin = np.min([ymin, np.mean(spks[ix, n])])
                            ymax = np.max([ymax, np.mean(spks[ix, n])])
                    
                    ### For each task ###
                    ix = (commands_disc[:, 0] == i_mag) & (commands_disc[:, 1] == i_ang) & (tsk == task)
                    ix = np.nonzero(ix==True)[0]
                    if len(ix) > 5:
                        axi.plot(np.mean(spks[ix, n]), np.mean(pred_spiking0[ix, n]), marker=marker[task], color='k', alpha=0.3)
                        axi.plot(np.mean(spks[ix, n]), np.mean(pred_spiking1[ix, n]), marker=marker[task], color='k', alpha=1.0)
                        axi.plot([np.mean(spks[ix, n]), np.mean(spks[ix, n])],
                            [np.mean(pred_spiking0[ix, n]), np.mean(pred_spiking1[ix, n])],
                            '-', color='k')
                    ymin = np.min([ymin, np.mean(spks[ix, n])])
                    ymax = np.max([ymax, np.mean(spks[ix, n])])
                axi.set_title('Mag %d, Ang %d' %(i_mag, i_ang))

                axi.plot([ymin, ymax], [ymin, ymax], 'k-', linewidth=.5)

        f.tight_layout()

### Given the model that explains y_t with a_t, how much variance remains? 
def plot_var_exp_action():
    mod = 'prespos_0psh_1spksm_0_spksp_0'
    animal = 'jeev'; model_set_number = 1; 
    model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %model_set_number, 'rb'))
    data = model_dict[0, mod]
    spks = model_dict[0, 'spks']

    resid = spks - data

    ### Population variance: 
    remaining_var = np.var(resid, axis=0) / np.var(spks, axis=0)

    ### Ok, now try fitting a model with different state lags plus action

def disc_plot(n_disc):

    fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw=dict(projection='polar'))

    # Generate some data...
    # Note that all of these are _2D_ arrays, so that we can use meshgrid
    # You'll need to "grid" your data to use pcolormesh if it's un-ordered points
    theta, r = np.mgrid[0:2*np.pi:9j, 0:4:5j]
    
    im1 = ax1.pcolormesh(theta, r, n_disc[:, :, 0].T, cmap = co_obs_cmap_cm[0], vmin=np.min(n_disc), vmax=np.max(n_disc))
    im2 = ax2.pcolormesh(theta, r, n_disc[:, :, 1].T, cmap = co_obs_cmap_cm[1], vmin=np.min(n_disc), vmax=np.max(n_disc))

    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    fig.savefig(fig_dir+'grom_day0_neur_38_targ4_targ5_dist.svg')

    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')

    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im2, cax=cax, orientation='vertical');
    # import pdb; pdb.set_trace()


#### Linear mixed effect modeling: 
def run_LME(Days, Grp, Metric):
    data = pd.DataFrame(dict(Days=Days, Grp=Grp, Metric=Metric))
    md = smf.mixedlm("Metric ~ Grp", data, groups=data["Days"])
    mdf = md.fit()
    pv = mdf.pvalues['Grp']
    slp = mdf.params['Grp']
    return pv, slp

def get_R2(y_true, y_pred, pop = True, ignore_nans = False):
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

def get_cov_diffs(ix_co, ix_ob, spks, diffs, method = 1, mult=1.):
    cov1 = np.cov(mult*10*spks[ix_co, :].T); 
    cov2 = np.cov(mult*10*spks[ix_ob, :].T); 

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

### Try to get to the bottom of the subspace overlap thing...
def test_so(): 
    animal = 'grom' 
    model_set_number = 1; i_d = 0; 
    model_dict = pickle.load(open('/Users/preeyakhanna/fa_analysis/tuning_models_'+animal+'_model_set%d.pkl' %(model_set_number), 'rb'))

    ### Get the model
    ### Get correlation b/w pred_spks and NP: 
    model = 'prespos_0psh_1spksm_0_spksp_0' 

    pred_y = model_dict[i_d, model]
    true_y = model_dict[i_d, 'spks']
    psh = np.hstack(( model_dict[i_d, 'np'], np.ones((len(model_dict[i_d, 'np']), 1)) ))

    C = np.squeeze(np.array(np.mat(np.linalg.lstsq(psh, pred_y)[0].T))) # 44 x 3; 
    eCov1 = np.dot(C, np.dot(np.cov(psh.T), C.T))
    eCov2 = np.cov(pred_y.T)
    
    ov1, _ = subspace_overlap.get_overlap(None, None, first_UUT=eCov1, second_UUT=eCov2, main=False)# main_thresh = 0.9)
    ov2, _ = subspace_overlap.get_overlap(None, None, first_UUT=eCov2, second_UUT=eCov1, main=False)# main_thresh = 0.9)
    
    ### Ok, so always 
    
