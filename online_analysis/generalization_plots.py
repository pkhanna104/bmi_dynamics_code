import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import os
import pandas

from online_analysis import util_fcns, generate_models, generate_models_list, generate_models_utils
from online_analysis import plot_generated_models, plot_pred_fr_diffs, plot_fr_diffs
import analysis_config

import statsmodels.formula.api as smf
import scipy.stats
from sklearn.linear_model import Ridge

#### Load mag_boundaries ####
mag_boundaries = pickle.load(open(analysis_config.data_params['mag_bound_file']))

class DataExtract(object):

    def __init__(self, animal, day_ix, model_set_number = 6, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0',
        nshuffs = 1000):

        ### Make sure the model_nm is in the model set number 
        model_var_list, _, _, _ = generate_models_list.get_model_var_list(model_set_number)
        mods = [mod[1] for mod in model_var_list]
        assert(model_nm in mods)

        self.animal = animal
        self.day_ix = day_ix 
        self.model_nm = model_nm
        self.model_set_number = model_set_number
        self.nshuffs = nshuffs
        self.loaded = False

    def load(self): 
        spks0, push0, tsk0, trg0, bin_num0, rev_bin_num0, move0, dat = util_fcns.get_data_from_shuff(self.animal, self.day_ix)
        spks0 = 10*spks0; 

        #### Get subsampled
        tm0, tm1 = generate_models.get_temp_spks_ix(dat['Data'])

        ### Get subsampled 
        self.spks = spks0[tm0, :]
        self.spks_tm1 = spks0[tm1, :]

        self.push = push0[tm0, :]
        self.push_tm1 = push0[tm1, :]

        self.command_bins = util_fcns.commands2bins([self.push], mag_boundaries, self.animal, self.day_ix, 
                                       vel_ix=[3, 5])[0]
        self.command_bins_tm1 = util_fcns.commands2bins([self.push_tm1], mag_boundaries, self.animal, self.day_ix, 
                                       vel_ix=[3, 5])[0]
        self.move = move0[tm0]
        self.move_tm1 = move0[tm1]

        ### Set the task here 
        self.task = np.zeros_like(self.move)
        self.task[self.move >= 10] = 1.

        if 'spksm_1' in self.model_nm:
            assert(np.all(bin_num0[tm0] > 0))
    
        self.bin_num = bin_num0[tm0]
        self.rev_bin_num = rev_bin_num0[tm0]

        ### Get number of neurons 
        self.nneur_flt = float(self.spks.shape[1])
        self.nneur = self.spks.shape[1]
        
        ###############################################
        ###### Get predicted spikes from the model ####
        ###############################################
        model_fname = analysis_config.config[self.animal+'_pref']+'tuning_models_'+self.animal+'_model_set'+str(self.model_set_number)+'_.pkl'
        model_dict = pickle.load(open(model_fname, 'rb'))
        pred_spks = model_dict[self.day_ix, self.model_nm]
        self.pred_spks = 10*pred_spks; 

        ### Make sure spks and sub_spks match -- using the same time indices ###
        assert(np.allclose(self.spks, 10*model_dict[self.day_ix, 'spks']))
        
        ###############################################
        ###### Get shuffled prediction of  spikes  ####
        ###############################################
        if self.nshuffs > 0:
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(self.animal, self.day_ix, self.model_nm, nshuffs = self.nshuffs, 
                testing_mode = False)
            self.pred_spks_shuffle = 10*pred_spks_shuffle; 
        
        self.loaded = True

    def load_null_roll(self): 
        pred, true = plot_generated_models.get_shuffled_data_pred_null_roll(self.animal, self.day_ix, self.model_nm, nshuffs = self.nshuffs,
            testing_mode = False)
        self.null_roll_pred = 10*pred; 
        self.null_roll_true = 10*true

    def load_null_roll_pot_shuff(self):
        pred, true = plot_generated_models.get_shuffled_data_pred_null_roll_pot_shuff(self.animal, self.day_ix, self.model_nm, nshuffs = self.nshuffs,
            testing_mode = False)
        self.null_roll_pot_beh_pred = 10*pred; 
        self.null_roll_pot_beh_true = 10*true

######### UTILS ##########
def stack_dict(d):
    for k in d.keys():
        d[k] = np.hstack((d[k]))
    return d

def plot_stars(ax, ymax, xval, pv):
    if pv < 0.001: 
        ax.text(xval, 1.2*np.max(ymax), '***', horizontalalignment='center')
    elif pv < 0.01: 
        ax.text(xval, 1.2*np.max(ymax), '**', horizontalalignment='center')
    elif pv < 0.05:
        ax.text(xval, 1.2*np.max(ymax), '*', horizontalalignment='center')
    else:
        ax.text(xval, 1.2*np.max(ymax), 'n.s.', horizontalalignment='center')

def r2(y_true, y_pred):
    SSE = np.sum(np.square(y_true - y_pred))
    SST = np.sum(np.square(y_true - np.mean(y_true, axis=0)[np.newaxis, :]))
    return 1 - (SSE/SST), 0

def err(y_true, y_pred):
    assert(y_true.shape == y_pred.shape)
    n = float(y_true.shape[1])
    return np.mean(np.linalg.norm(y_true-y_pred, axis=1)/n), np.std(np.linalg.norm(y_true-y_pred, axis=1)/n)

def cc(x, y):
    _,_,rv,_,_ = scipy.stats.linregress(x, y)
    return rv

def get_spks(animal, day_ix):
    spks0, push, _, _, _, _, move, dat = util_fcns.get_data_from_shuff(animal, day_ix)
    spks0 = 10*spks0; 

     #### Get subsampled
    tm0, tm1 = generate_models.get_temp_spks_ix(dat['Data'])

    ### Get subsampled 
    spks_sub = spks0[tm0, :]
    push_sub = push[tm0, :]
    push_sub_tm1 = push[tm1, :]
    move_sub = move[tm0]

    ### Get command bins 
    command_bins = util_fcns.commands2bins([push_sub], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]
    command_bins_tm1 = util_fcns.commands2bins([push_sub_tm1], mag_boundaries, animal, day_ix, 
                                       vel_ix=[3, 5])[0]

    return spks_sub, command_bins, move_sub, push_sub, command_bins_tm1

def lo_in_key(lo_key, lo_val, cat):
        """check where the left out value (lo_val)
        corresponds to the key of the saved predictions (lo_key)
        
        Parameters
        ----------
        lo_key : tuple? 
            key of saved preditions 
        lo_val : str
            value corresponding to left out thing 
        cat : str
            category of left out thing (tsk/mov/com)
        
        Returns
        -------
        TYPE
            Description
        """
        in_key = False 

        if cat == 'tsk':
            if lo_key[2] >= 10. and int(lo_val) == 1:
                in_key = True
            elif lo_key[2] < 10. and int(lo_val) == 0:
                in_key = True
        elif cat == 'mov':
            if lo_key[2] == lo_val:
                in_key = True
        elif cat == 'com':
            if lo_key[0]*8 + lo_key[1] == lo_val:
                in_key = True

        return in_key

def check_ix_lo(lo_ix, com_true, com_true_tm1, mov_true, left_out, cat):
    
    if cat == 'tsk':
        if left_out == 0.:  
            assert(np.all(mov_true[lo_ix] < 10))
        elif left_out == 1.:
            assert(np.all(mov_true[lo_ix] >= 10.))
    
    elif cat == 'mov':
        assert(np.all(mov_true[lo_ix] == left_out))
    
    elif cat == 'com':
        vl0 = com_true[lo_ix, 0]*8+com_true[lo_ix,1]==left_out
        vl1 = com_true_tm1[lo_ix, 0]*8+com_true_tm1[lo_ix,1]==left_out
        vl2 = np.logical_or(vl0, vl1)
        assert(np.all(vl2))

########## model fitting utls ###########
def train_and_pred(spks_tm1, spks, push, train, test, alpha, KG,
    add_mean = None, skip_cond = False):
    """Summary
    
    Parameters
    ----------
    spks_tm1 : TYPE
        Description
    spks : TYPE
        Description
    push : TYPE
        Description
    train : TYPE
        np.array of training indicies 
    test : TYPE
        np.array of test indices 
    alpha : TYPE
        ridge parameter for regression for this day from swept alphas. 
    KG : TYPE
        Description
    add_mean : None, optional
         add_mean = [mean_spks_ix, mean_spks_true]
    
    Deleted Parameters
    ------------------
    data_temp : TYPE
        dictonary from get_spike_kinematics 
    nneur : int
        number of neurons
    
    Returns
    -------
    TYPE
        Description
    
    """
    X_train = spks_tm1[train, :]
    X_test = spks_tm1[test, :]
    push_test = push[test, :]
    y_train = spks[train, :]

    model = Ridge(alpha=alpha, fit_intercept=True)

    ### Fit the model: args = X, y
    model.fit(X_train, y_train)

    ### Estimate error covariance; 
    y_train_est = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if add_mean:
        cnt = 0 
        for i, (ix, mn) in enumerate(zip(add_mean[0], add_mean[1])):
            ixx = np.array([ji for ji, j in enumerate(test) if j in ix]); 
            y_test_pred[ixx, :] = y_test_pred[ixx, :] + mn[np.newaxis, :]
            cnt += len(ixx)
        assert(cnt == len(test))

    if skip_cond:
        return y_test_pred, model.coef_, model.intercept_
    else:
        return add_cond(y_train, y_train_est, X_train, KG, y_test_pred, push_test, model)

def add_cond(y_train, y_train_est, X_train, KG, y_test_pred, push_test, model):
    ### Row is a variable, column is an observation for np.cov
    W = np.cov((y_train - y_train_est).T)
    assert(W.shape[0] == X_train.shape[1])

    ### Now estimate held out data with conditioning ###
    cov12 = np.dot(KG, W).T
    cov21 = np.dot(KG, W)
    cov22 = np.dot(KG, np.dot(W, KG.T))
    cov22I = np.linalg.inv(cov22)

    T = y_test_pred.shape[0]

    pred_w_cond = []
    
    for i_t in range(T):

        ### Get this prediction (mu 1)
        mu1_i = y_test_pred[i_t, :][:, np.newaxis]

        ### Get predicted value of action; 
        mu2_i = np.dot(KG, mu1_i)

        ### Actual action; 
        a_i = push_test[i_t, :][:, np.newaxis]

        ### Conditon step; 
        mu1_2_i = mu1_i + np.dot(cov12, np.dot(cov22I, a_i - mu2_i))

        ### Make sure it matches; 
        assert(np.allclose(np.dot(KG, mu1_2_i), a_i))

        pred_w_cond.append(np.squeeze(np.array(mu1_2_i)))

    return np.vstack((pred_w_cond)), model.coef_, model.intercept_

class Mod(object):
    def __init__(self, A, b):
        self.coef_ = A 
        self.intercept_ = b

def train_spec_b(spks_tm1, spks, push, train, test, KG, Agen):
    X_train = spks_tm1[train, :]
    X_test = spks_tm1[test, :]
    
    push_test = push[test, :]
    y_train = spks[train, :]

    ### Estimate y_t - Ay_{t-1}
    rez = y_train - np.dot(Agen, X_train.T).T
    B = np.mean(rez, axis=0) 
    y_train_est = np.dot(Agen, X_train.T).T + B[np.newaxis, :]
    y_test_pred = np.dot(Agen, X_test.T).T + B[np.newaxis, :]

    model = Mod(Agen, B)

    return add_cond(y_train, y_train_est, X_train, KG, y_test_pred, push_test, model)

def get_com(tby2_push, animal, day_ix):
    command_bins = util_fcns.commands2bins([tby2_push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[0, 1])[0]
    return command_bins[:, 0]*8 + command_bins[:, 1], command_bins[:, 0], command_bins[:, 1]

def save_ax_cc_err(ax, ax_err, f, f_err, cat):
    ax.set_xlim([-1, 14])
    ax.set_xticks([])
    ax.set_ylabel('Corr Coeff')

    ax_err.set_xlim([-1, 14])
    ax_err.set_xticks([])
    ax_err.set_ylabel('Err')

    f.tight_layout()
    f_err.tight_layout()

    util_fcns.savefig(f, '%s_lo_pop_dist_rv'%cat)
    util_fcns.savefig(f_err, '%s_lo_pop_dist_err'%cat)
    
######## Bar plots by task / target / command ##########
def plot_err_by_cat(model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', yval = 'err',
    run_f_test = True): 
    
    if yval == 'err':
        yfcn = err; 
    elif yval == 'r2':
        yfcn = r2; 

    '''
    Plot the R2 of each category of movement 
    '''
    ftsk, axtsk = plt.subplots(ncols = 5, nrows = 3, figsize = (15, 10))
    fmov, axmov = plt.subplots(ncols = 5, nrows = 3, figsize = (15, 10))
    fcom, axcom = plt.subplots(ncols = 5, nrows = 3, figsize = (15, 10))

    r2_tsk = dict(r2=[], tsk=[], anim_day=[])
    r2_mov = dict(r2=[], mov=[], anim_day=[])
    r2_com = dict(r2=[], com=[], anim_day=[])

    fN, axN = plt.subplots()
    axN.set_xlabel('# Observations')
    axN.set_ylabel('r2')
    N_dat = []

    for i_a, animal in enumerate(['grom', 'jeev']): 

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ri = (i_a*10 + day_ix) / 5
            ci = (i_a*10 + day_ix) % 5

            ### Load data ###
            dataObj = DataExtract(animal, day_ix, model_nm = model_nm, nshuffs=0)
            dataObj.load()

            ############# Task  #############
            ### Now lets do R2 by task first ###
            for task in range(2): 

                ### Indices 
                ix = np.nonzero(dataObj.task == task)[0]

                ### Get Err
                mn, sd = yfcn(dataObj.spks[ix, :], dataObj.pred_spks[ix, :])
                r2_tsk['r2'].append(mn)
                r2_tsk['tsk'].append(task)
                r2_tsk['anim_day'].append(i_a*10 + day_ix)

                axtsk[ri, ci].bar(task, mn, color='k', alpha=0.2)
                axtsk[ri, ci].errorbar(task, mn, sd, color='k', marker='|')
                axN.plot(len(ix), mn, 'k.')
                N_dat.append([len(ix), mn])

            axtsk[ri, ci].set_title('%s, %d'%(animal, day_ix))

            ############# Movements #############
            for i_m, mov in enumerate(np.unique(dataObj.move)):

                ### Indices 
                ix = np.nonzero(dataObj.move == mov)[0]

                ### Get R2 
                mn, sd = yfcn(dataObj.spks[ix, :], dataObj.pred_spks[ix, :])
                r2_mov['r2'].append(mn)
                r2_mov['mov'].append(mov)
                r2_mov['anim_day'].append(i_a*10 + day_ix)  

                axmov[ri, ci].bar(i_m, mn, color=util_fcns.get_color(mov))
                axmov[ri, ci].errorbar(i_m, mn, sd, color='k', marker='|')
                axN.plot(len(ix), mn, 'b.')
                N_dat.append([len(ix), mn])
            axmov[ri, ci].set_title('%s, %d'%(animal, day_ix))
            
            ############# Commands #############            
            for mag in range(4):
                for ang in range(8):

                    ix = np.nonzero(np.logical_and(dataObj.command_bins[:, 0] == mag, dataObj.command_bins[:, 1] == ang))[0]

                    if len(ix) > 0:

                        ### Get R2 
                        mn, sd = yfcn(dataObj.spks[ix, :], dataObj.pred_spks[ix, :])
                        r2_com['r2'].append(mn)
                        r2_com['com'].append(mag*10 + ang)
                        r2_com['anim_day'].append(i_a*10 + day_ix)  
                        axcom[ri, ci].bar(mag*8 + ang, mn, color=util_fcns.get_color(np.mod(ang + 3, 8)))
                        axcom[ri, ci].errorbar(mag*8 + ang, mn, sd, color='k', marker='|')
                        axN.plot(len(ix), mn, 'r.')
                        N_dat.append([len(ix), mn])
            axcom[ri, ci].set_title('%s, %d'%(animal, day_ix))

    
    for f in [ftsk, fmov, fcom, fN]:
        f.tight_layout()

    N_dat = np.vstack((N_dat))
    N_dat[np.isinf(N_dat[:, 1]), 1] = np.nan
    ixtoss = np.nonzero(np.isnan(N_dat[:, 1])==True)[0]
    ixkeep = np.array([i for i in range(len(N_dat)) if i not in ixtoss])
    slp,intc,rv,pv,err = scipy.stats.linregress(N_dat[ixkeep, 0], N_dat[ixkeep, 1])
    x_ = np.linspace(np.min(N_dat[:, 0]), np.max(N_dat[:, 0]), 10)
    y_ = slp*x_ + intc; 
    axN.plot(x_, y_, 'k--')
    axN.set_title('r=%.2f, pv=%.4f' %(rv, pv))

    import pdb; pdb.set_trace()

    if run_f_test:
        ###### Plot the models ########
        r2_tsk = stack_dict(r2_tsk)
        r2_mov = stack_dict(r2_mov)
        r2_com = stack_dict(r2_com)

        ########### Task ##########
        ymax = []
        for tsk in range(2):
            ix = np.nonzero(r2_tsk['tsk'] == tsk)[0]
            #axtsk.bar(tsk, np.mean(r2_tsk['r2'][ix]), .8, color='k', alpha=0.2)
            ymax.append(np.mean(r2_tsk['r2'][ix]))
        md = smf.mixedlm("r2 ~ C(tsk)", r2_tsk, groups=r2_tsk['anim_day'])
        mdf = md.fit()
        assert(len(mdf.params) == 2 + 1) ## 2 tsks - 1 + intc + grp = 
        A = np.identity(len(mdf.params))
        A = A[1:-1, :] ### Remove first row (intercept) and last row (gropu)
        assert(A.shape[0] == A.shape[1] - 2)
        rez = mdf.f_test(A)
        
        pv_tsk = rez.pvalue; 
        #axtsk.plot([0, 1], [1.1*np.max(ymax), 1.1*np.max(ymax)], '-')
        #plot_stars(axtsk, ymax, .5, pv_tsk)
            
        ########### Mov ##########
        ymax = []
        Nmov = len(np.unique(r2_mov['mov']))
        for i_m, mov in enumerate(np.unique(r2_mov['mov'])):
            ix = np.nonzero(r2_mov['mov'] == mov)[0]
            #axmov.bar(i_m, np.mean(r2_mov['r2'][ix]), .8, color=util_fcns.get_color(mov))
            ymax.append(np.mean(r2_mov['r2'][ix]))

        md = smf.mixedlm("r2 ~ C(mov)", r2_mov, groups=r2_mov['anim_day'])
        mdf = md.fit()
        assert(len(mdf.params) == Nmov + 1)
        
        A = np.identity(len(mdf.params))
        A = A[1:-1, :] ### Remove first row and last row 
        assert(A.shape[0] == A.shape[1] - 2)
        rez = mdf.f_test(A)
        
        pv_mov = rez.pvalue; 
        #axmov.plot([0, Nmov], [1.1*np.max(ymax), 1.1*np.max(ymax)], '-')
        #plot_stars(axmov, ymax, .5*Nmov, pv_mov)

        ########### Command ##########
        ymax = []
        Ncom = len(np.unique(r2_com['com']))
        for i_c, com in enumerate(np.unique(r2_com['com'])):
            ix = np.nonzero(r2_com['com'] == com)[0]
            #axcom.bar(i_c, np.mean(r2_com['r2'][ix]), .8, color='k', alpha=0.2)
            ymax.append(np.mean(r2_com['r2'][ix]))

        md = smf.mixedlm("r2 ~ C(com)", r2_com, groups=r2_com['anim_day'])
        mdf = md.fit()
        assert(len(mdf.params) == Ncom + 1)
        
        A = np.identity(len(mdf.params))
        A = A[1:-1, :] ### Remove first row 
        rez = mdf.f_test(A)
        
        pv_com = rez.pvalue; 
        #axcom.plot([0, Ncom], [1.1*np.max(ymax), 1.1*np.max(ymax)], '-')
        #plot_stars(axcom, ymax, .5*Ncom, pv_com)

######## Fit general model ############
def fit_predict_loo_model(cat='tsk', mean_sub_tsk_spec = False, 
    n_folds = 5, min_num_per_cat_lo = 15, model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', 
    model_set_number = 6, match_command_dist = False, zero_alpha=False, tsk_spec_alphas=False):

    """Summary
    
    Parameters
    ----------
    cat : str, optional
        Description
    """
    cat_dict = {}
    cat_dict['tsk'] = [['tsk'], np.arange(2)]
    cat_dict['mov'] = [['mov'], np.unique(np.hstack((np.arange(8), np.arange(10, 20), np.arange(10, 20)+.1)))]
    cat_dict['com'] = [['com', 'com1'], np.arange(32)]
    cat_dict['com_ang'] = [['com_ang', 'com_ang1'], np.arange(8)]
    cat_dict['com_mag'] = [['com_mag', 'com_mag1'], np.arange(4)]


    #### which category are we dealing with ####
    leave_out_fields = cat_dict[cat][0]
    leave_out_cats = cat_dict[cat][1]

    if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
        skip_cond = False
    elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0':
        skip_cond = True

    ### get the right alphase for the model ####
    if tsk_spec_alphas:
        ridge_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'], 'tsk_spec_alphas.pkl'), 'rb')); 
    else:
        ridge_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'] , 'max_alphas_ridge_model_set%d.pkl'%model_set_number), 'rb')); 

    for i_a, animal in enumerate(['grom', 'jeev']):

        input_type = analysis_config.data_params['%s_input_type'%animal]
        ord_input_type = analysis_config.data_params['%s_ordered_input_type'%animal]

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            print('Starting %s, Day %d' %(animal, day_ix))

            #### Get ridge alpha ####
            if zero_alpha:
                alpha_spec = 1.
            elif tsk_spec_alphas:
                #### Alphas if fitting on Co / OBS = [ridge_dict[animal, day_ix, 0], ridge_dict[animal, day_ix, 1]] 
                #### Alphase if leaving out CO / OBS = [ridge_dict[animal, day_ix, 1], ridge_dict[animal, day_ix, 0]] 
                alpha_spec_list = [ridge_dict[animal, day_ix, 1], ridge_dict[animal, day_ix, 0]] 
            else:
                alpha_spec = ridge_dict[animal][0][day_ix, model_nm]

            KG = util_fcns.get_decoder(animal, day_ix)

            #### Get data ####
            data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, input_type[day_ix], 
            ord_input_type[day_ix], 1, within_bin_shuffle = False, day_ix = day_ix, skip_plot = True)

            ############## Data checking ##############
            nneur = sub_spikes.shape[1]
            for n in range(nneur):
                assert(np.allclose(sub_spk_temp_all[:, 0, n], data_temp['spk_tm1_n%d'%n]))
                assert(np.allclose(sub_spk_temp_all[:, 1, n], data_temp['spk_tm0_n%d'%n]))

            push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T
            push_tm1 = np.vstack((data_temp['pshx_tm1'], data_temp['pshy_tm1'])).T


            #### Add teh movement category ###
            data_temp['mov'] = data_temp['trg'] + 10*data_temp['tsk']
            data_temp['com'], data_temp['com_mag'], data_temp['com_ang'] = get_com(sub_push_all, animal, day_ix)
            data_temp['com1'], data_temp['com_mag1'], data_temp['com_ang1'] = get_com(push_tm1, animal, day_ix)            
            
            ############## Get subspikes ##############
            spks_tm1 = sub_spk_temp_all[:, 0, :]
            spks_tm0 = sub_spk_temp_all[:, 1, :]
            
            LOO_dict = {}
            LOO_dict_ctrl = {}

            ############# Estimate command dist diffs?  ###############
            if match_command_dist:
                ix_tsk0 = np.nonzero(data_temp['tsk']==0)[0]
                ix_tsk1 = np.nonzero(data_temp['tsk']==1)[0]

                ############# Match the command distributions #############
                ix_keep1, ix_keep2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_tm0[ix_tsk0,:], push_tm0[ix_tsk1, :], perc_drop = 0.01)
                analyze_indices = np.hstack(( ix_tsk0[ix_keep1], ix_tsk1[ix_keep2] ))

                ############ Re-index everything ###########################
                data_temp = subsampPd(data_temp, analyze_indices)
                spks_tm0 = spks_tm0[analyze_indices, :]
                spks_tm1 = spks_tm1[analyze_indices, :]
                push_tm0 = push_tm0[analyze_indices, :]
                LOO_dict['analyze_indices'] = analyze_indices
                LOO_dict_ctrl['analyze_indices'] = analyze_indices

            ############ Mean subtraction -- task specifci ######
            if mean_sub_tsk_spec:
                mean_spks_true = []; mean_spks_ix = []; 
                for i in range(2):
                    ix0 = np.nonzero(data_temp['tsk']==i)[0]

                    ### Make the means the same 
                    mean_spks_true0 = np.mean(spks_tm0, axis=0)

                    #mean_spks_true0 = np.mean(spks_tm0[ix0, :], axis=0)
                    spks_tm0[ix0, :] = spks_tm0[ix0, :] - mean_spks_true0[np.newaxis, :]
                    spks_tm1[ix0, :] = spks_tm1[ix0, :] - mean_spks_true0[np.newaxis, :]
                    mean_spks_true.append(mean_spks_true0)
                    mean_spks_ix.append(ix0)
                add_spks = [mean_spks_ix, mean_spks_true]
            
            else:
                add_spks = None

            #### Make 5 folds --> all must exclude the thing you want to exclude, but distribute over the test set; 
            test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)

            #### setup data storage 
            for lo in leave_out_cats: 
    
                ix_rm = []
                for leave_out_field in leave_out_fields:
                    #### Which indices must be removed from all training sets? 
                    ix_rm.append(np.nonzero(data_temp[leave_out_field] == lo)[0])
                ix_rm = np.hstack((ix_rm))

                N = len(data_temp[leave_out_field])
                n = len(ix_rm)
                ix_rm_rand = np.random.permutation(N)[:n]

                if len(ix_rm) >= min_num_per_cat_lo: 

                    LOO_dict[lo] = {}
                    LOO_dict_ctrl[lo] = {}

                    y_pred_lo  = np.zeros_like(spks_tm0) + np.nan
                    y_pred_nlo = np.zeros_like(spks_tm0) + np.nan

                    if tsk_spec_alphas:
                        alpha_spec = alpha_spec_list[int(lo)]

                    #### Go through the train_ix and remove the particualr thing; 
                    for i_fold in range(n_folds):
                        
                        train_fold_full = train_ix[i_fold]
                        test_fold  = test_ix[i_fold]

                        #### removes indices 
                        train_fold_lo = np.array([ i for i in train_fold_full if i not in ix_rm])
                        
                        print('Starting fold %d for LO %1.f, training pts = %d, alpha %.1f' %(i_fold, lo, len(train_fold_lo), alpha_spec))

                        #### randomly remove indices 
                        train_fold_nlo = np.array([i for i in train_fold_full if i not in ix_rm_rand])

                        #### train the model and predict held-out data ###
                        y_pred_lo[test_fold, :], coef, intc = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_fold_lo, test_fold, alpha_spec, KG, add_mean = add_spks, skip_cond=skip_cond)

                        ### Train the model and predict held out data -- but subselect so that 
                        ### Add these predictions to test_fold: 
                        ### The idea here is to create a normal dynamics prediction that has used the 
                        ### same amount of training data; 
                        test_fold_nlo = np.sort(np.hstack((test_fold, ix_rm_rand)))
                        y_pred_nlo[test_fold_nlo, :], coef_nlo, intc_nlo = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_fold_nlo, test_fold_nlo, alpha_spec, KG, add_mean = add_spks, skip_cond=skip_cond)
                        
                        ### Save the matrices; 
                        LOO_dict[lo][i_fold, 'coef_lo'] = coef
                        LOO_dict[lo][i_fold, 'intc_lo'] = intc
                        LOO_dict_ctrl[lo][i_fold, 'coef_nlo'] = coef_nlo
                        LOO_dict_ctrl[lo][i_fold, 'intc_nlo'] = intc_nlo

                    #### Now go through and compute movement-specific commands ###
                    assert(np.sum(np.isnan(y_pred_lo)) == 0)
                    assert(np.sum(np.isnan(y_pred_nlo)) == 0)
                        
                    for mag in range(4):
                        for ang in range(8):
                            for mov in np.unique(data_temp['mov']):
                                mc = np.where((data_temp['com'] == mag*8 + ang) & (data_temp['mov'] == mov))
                                assert(type(mc) is tuple)
                                mc = mc[0]
                                if len(mc) >= 15:
                                    LOO_dict[lo][mag, ang, mov] = y_pred_lo[mc, :].copy()
                                    LOO_dict[lo][mag, ang, mov, 'ix'] = mc.copy()

                    #### Also generally save the indices of the thing you left out of training; 
                    LOO_dict[lo][-1, -1, -1] = y_pred_lo[ix_rm, :].copy()
                    LOO_dict[lo][-1, -1, -1, 'ix'] = ix_rm.copy()

                    ### otherwise sampled data ####
                    LOO_dict_ctrl[lo]['y_pred_nlo'] = y_pred_nlo.copy()

                #### Save this LOO ####
            if mean_sub_tsk_spec:
                ext1 = '_tsksub'
            else:
                ext1 = ''

            if zero_alpha:
                ext2 = '_zeroalph'
            elif tsk_spec_alphas:
                ext2 = '_tskalph'
            else:
                ext2 = ''

            if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
                ext3 = ''
            elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0':
                ext3 = '_nocond'

            pickle.dump(LOO_dict, open(os.path.join(analysis_config.config['grom_pref'], 'loo_%s_%s_%d%s%s%s.pkl'%(cat, animal, day_ix, ext1, ext2, ext3)), 'wb'))
            pickle.dump(LOO_dict_ctrl, open(os.path.join(analysis_config.config['grom_pref'], 'loo_ctrl_%s_%s_%d%s%s%s.pkl'%(cat, animal, day_ix, ext1, ext2, ext3)), 'wb'))

######## Fit move group model #####
def fit_predict_lomov_model(min_num_per_cat_lo = 15, 
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', model_set_number = 6, nshuffs=10,
    min_commands = 15):

    """Summary
    
    Parameters
    ----------
    cat : str, optional
        Description
    """
    cat_dict = {}
    cat_dict['vert']     = dict(grom=[1., 5., 11.0, 11.1, 15.0, 15.1], jeev=[2.0, 6.0, 12.0, 12.1, 13.0, 13.1, 14.0, 14.1, 15.0, 15.1])
    cat_dict['horz']     = dict(grom=[3., 7., 13.0, 13.1, 17.0, 17.1], jeev=[0.0, 4.0, 18.0, 18.1, 19.0, 19.1])
    cat_dict['diag_pos'] = dict(grom=[0., 4., 10.0, 10.1, 14.0, 14.1], jeev=[1.0, 5.0, 16.0, 16.1])
    cat_dict['diag_neg'] = dict(grom=[2., 6., 12.0, 12.1, 16.0, 16.1])

    ridge_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'] , 'max_alphas_ridge_model_set%d.pkl'%model_set_number), 'rb')); 

    for cat_ in cat_dict.keys(): 

        animals = np.sort(cat_dict[cat_].keys())
        f, ax = plt.subplots(figsize=(3, 3))
        ax.set_title(cat_)

        f_cc, ax_cc = plt.subplots(figsize=(2, 3))
        f_err, ax_err = plt.subplots(figsize=(2, 3))


        ##### cycle through the animals for this ###
        for i_a, animal in enumerate(animals):

            input_type = analysis_config.data_params['%s_input_type'%animal]
            ord_input_type = analysis_config.data_params['%s_ordered_input_type'%animal]
            mov_train = cat_dict[cat_][animal]

            model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(6)+'_.pkl'
            model_dict = pickle.load(open(model_fname, 'rb'))
            model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'

            for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

                print('Starting %s, Day %d Mov Cat %s' %(animal, day_ix, cat_))

                alpha_spec = ridge_dict[animal][0][day_ix, model_nm]

                KG = util_fcns.get_decoder(animal, day_ix)

                # #### Get data ####
                data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, input_type[day_ix], 
                    ord_input_type[day_ix], 1, within_bin_shuffle = False, day_ix = day_ix, skip_plot = True)

                ### Load predicted data freom the model 
                sub_spikes = 10*sub_spikes
                spks_pred = 10*model_dict[day_ix, model_nm]
                spks_pred_shuffle = 10*plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                    testing_mode = False)

                # ############## Data checking ##############
                nneur = sub_spikes.shape[1]
                for n in range(nneur):
                    assert(np.allclose(sub_spk_temp_all[:, 0, n], data_temp['spk_tm1_n%d'%n]))
                    assert(np.allclose(sub_spk_temp_all[:, 1, n], data_temp['spk_tm0_n%d'%n]))

                push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T
                command_bins = util_fcns.commands2bins([push_tm0], mag_boundaries, animal, day_ix,
                    vel_ix=[0, 1])[0]

                # #### Add teh movement category ###
                data_temp['mov'] = data_temp['trg'] + 10*data_temp['tsk']
                
                ############## Get subspikes ##############
                spks_tm1 = 10*sub_spk_temp_all[:, 0, :]
                spks_tm0 = 10*sub_spk_temp_all[:, 1, :]
                
                LOO_dict = {}
                LOO_dict_ctrl = {}

                train_ix = []; N = push_tm0.shape[0]
                for mt in mov_train:
                    ix_mt = np.nonzero(data_temp['mov'] == mt)[0]
                    if len(ix_mt) > 0:
                        train_ix.append(ix_mt)

                ##### Train / Test #####
                train_ix = np.sort(np.hstack((train_ix)))
                test_ix = np.array([i for i in range(N) if i not in train_ix])
                
                ##### Model fitting; #####
                y_lo_pred, _, _ = train_and_pred(0.1*spks_tm1, 0.1*spks_tm0, push_tm0,
                    train_ix, test_ix, alpha_spec, KG)

                ### Multipy by 10 to match the spks
                y_lo_pred_10 = y_lo_pred*10; 
                
                y_std_pred = spks_pred[test_ix, :]

                r2_lo = util_fcns.get_R2(0.1*spks_tm0[test_ix, :], y_lo_pred)
                r2_std = util_fcns.get_R2(0.1*spks_tm0[test_ix, :], 0.1*y_std_pred)
                ax.plot(i_a*10 + day_ix - 0.1, r2_lo, '.', color='purple')
                ax.plot(i_a*10 + day_ix + 0.1, r2_std, '.', color=analysis_config.blue_rgb)
                
                r2_shuff = []
                for i in range(nshuffs):
                    y_shuf = spks_pred_shuffle[test_ix, :, i]
                    r2_shuff.append(util_fcns.get_R2(0.1*spks_tm0[test_ix, :], 0.1*y_shuf))
                util_fcns.draw_plot(i_a*10 + day_ix, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
                ax.plot([i_a*10 + day_ix, i_a*10 + day_ix], [np.mean(r2_shuff), np.max([r2_lo, r2_std])],
                    'k-', linewidth=0.5)

                ###### Also predict the pairwise issues; #####
                ###### Go through each unique movement in test_ix and compare it to mov2 from pred_spks ####
                unique_mov_lo = np.unique(data_temp['mov'][test_ix])
                unique_mov = np.unique(data_temp['mov'])

                ###### Look at differences #####
                True_diffs = []
                LO_pred_diffs = []
                Pred_diffs = []
                shuff_diff = {}
                for i in range(nshuffs): shuff_diff[i] = []

                #### Now for each command? 
                for mag in range(4):

                    for ang in range(8): 

                        #### Get command indices ####
                        ix_com_lo =  (command_bins[test_ix, 0] == mag) & (command_bins[test_ix, 1] == ang)
                        ix_com_lo = np.nonzero(ix_com_lo)[0]

                        ix_com_all = (command_bins[:, 0]       == mag) & (command_bins[:, 1]       == ang)
                        ix_com_all = np.nonzero(ix_com_all)[0]

                        #### Get movements 1 #####
                        for i_m1, mov1 in enumerate(unique_mov_lo): 
                            ix_mov1 = np.nonzero(data_temp['mov'][test_ix[ix_com_lo]] == mov1)[0]
                            ix_mov1_pred = np.nonzero(data_temp['mov'][ix_com_all] == mov1)[0]

                            ##### Get movements 2 ######
                            for i_m2, mov2 in enumerate(unique_mov):

                                if mov1 != mov2: 
    
                                    ix_mov2 = np.nonzero(data_temp['mov'][ix_com_all] == mov2)[0]

                                    ##### Make sure these are greater than minimum commands #####
                                    if len(ix_mov1) >= min_commands and len(ix_mov2) >= min_commands:
                                
                                        ##### Match the distributions ####
                                        ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_tm0[test_ix[ix_com_lo[ix_mov1]], :],
                                            push_tm0[ix_com_all[ix_mov2], :], psig=.05)

                                        ##### If these match ###
                                        if len(ix1_1) >= min_commands and len(ix2_2) >= min_commands:

                                            #### Get these indices ####
                                            mov1_ix_LO = ix_com_lo[ix_mov1[ix1_1]] ##### Keep the test_ix out, since that's already addumed in y_lo_pred
                                            mov1_ix = ix_com_all[ix_mov1_pred[ix1_1]]
                                            mov2_ix = ix_com_all[ix_mov2[ix2_2]]

                                            assert(np.all(command_bins[test_ix[mov1_ix_LO], 0] == mag))
                                            assert(np.all(command_bins[test_ix[mov1_ix_LO], 1] == ang))
                                            assert(np.all(data_temp['mov'][test_ix[mov1_ix_LO]] == mov1))

                                            assert(np.all(command_bins[mov1_ix, 0] == mag))
                                            assert(np.all(command_bins[mov1_ix, 1] == ang))
                                            assert(np.all(data_temp['mov'][mov1_ix] == mov1))

                                            assert(np.all(command_bins[mov2_ix, 0] == mag))
                                            assert(np.all(command_bins[mov2_ix, 1] == ang))
                                            assert(np.all(data_temp['mov'][mov2_ix] == mov2))
                                            
                                            m1_left_out_mean = np.mean(y_lo_pred_10[mov1_ix_LO, :], axis=0)

                                            m1_pred_mean = np.mean(spks_pred[mov1_ix, :], axis=0)
                                            m2_pred_mean = np.mean(spks_pred[mov2_ix, :], axis=0)
                                            
                                            m1_true_mean = np.mean(sub_spikes[mov1_ix, :], axis=0)
                                            m2_true_mean = np.mean(sub_spikes[mov2_ix, :], axis=0)

                                            m1_shuff_pred_mean = np.mean(spks_pred_shuffle[mov1_ix, :, :], axis=0)
                                            m2_shuff_pred_mean = np.mean(spks_pred_shuffle[mov2_ix, :, :], axis=0)


                                            True_diffs.append(plot_pred_fr_diffs.nplanm(m1_true_mean, m2_true_mean))
                                            Pred_diffs.append(plot_pred_fr_diffs.nplanm(m1_pred_mean, m2_pred_mean))
                                            LO_pred_diffs.append(plot_pred_fr_diffs.nplanm(m1_left_out_mean, m2_true_mean))

                                            for n in range(nshuffs):
                                                shuff_diff[n].append(plot_pred_fr_diffs.nplanm(m1_shuff_pred_mean[:, n], m2_shuff_pred_mean[:, n]))

                #### For each movement, comparison b/w LO commadn estimate for a given movement vs. predicted 
                plot_LO_means(True_diffs, Pred_diffs, LO_pred_diffs, shuff_diff, ax_cc, ax_err, nshuffs, i_a*10 + day_ix,
                    animal, day_ix, title_str='move_dir_loo_%s'%cat_)

            save_ax_cc_err(ax_cc, ax_err, f_cc, f_err, 'move_dir_loo_%s'%cat_)

        ##### Plotting #####
        ax.set_xlim([-1, 14])
        ax.set_xticks([])
        f.tight_layout()
        util_fcns.savefig(f, 'mov_grp_train_%s'%cat_)
      
def get_tsk_spec_alpha(n_folds=5):
    alphas = [np.arange(10, 100, 10), np.arange(100, 1000, 100), np.arange(1000, 10000, 1000)]; 
    # for i in range(-4, 7):
    #     alphas.append((1./4)*10**i)
    #     alphas.append((2./4.)*10**i)
    #     alphas.append((3./4.)*10**i)
    #     alphas.append(1.*10**i)
    alphas = np.hstack((alphas))
    alphas = alphas.astype(float)

    max_alpha = {}

    for i_a, animal in enumerate(['grom', 'jeev']):

        input_type = analysis_config.data_params['%s_input_type'%animal]
        ord_input_type = analysis_config.data_params['%s_ordered_input_type'%animal]

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            print('Starting %s, Day %d' %(animal, day_ix))
            KG = util_fcns.get_decoder(animal, day_ix)

            #### Get data ####
            data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, input_type[day_ix], 
            ord_input_type[day_ix], 1, within_bin_shuffle = False, day_ix = day_ix, skip_plot = True)
            
            ############## Data checking ##############
            nneur = sub_spikes.shape[1]
            for n in range(nneur):
                assert(np.allclose(sub_spk_temp_all[:, 0, n], data_temp['spk_tm1_n%d'%n]))
                assert(np.allclose(sub_spk_temp_all[:, 1, n], data_temp['spk_tm0_n%d'%n]))
            
            ############## Get subspikes ##############
            push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T
            spks_tm1 = sub_spk_temp_all[:, 0, :]
            spks_tm0 = sub_spk_temp_all[:, 1, :]
            
            ### Specifically for task ####
            for tsk in [0, 1]: 
                r2_alph = np.zeros_like(alphas)

                #### Which indices must be removed from all training sets? 
                ix_keep = np.nonzero(data_temp['tsk'] == tsk)[0]
   
                #### Make 5 folds --> all must exclude the thing you want to exclude, but distribute over the test set; 
                test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, subsampPd(data_temp, ix_keep))
                push_tm0_tsk = push_tm0[ix_keep, :]
                spks_tm1_tsk = spks_tm1[ix_keep, :]
                spks_tm0_tsk = spks_tm0[ix_keep, :]
                
                for ialpha, alpha_spec in enumerate(alphas):
                    
                    y_pred_lo  = np.zeros_like(spks_tm0_tsk) + np.nan
                    print('Starting alpha%.1f tsk %1.f, training pts = %d' %(alpha_spec, tsk, len(train_ix[0])))
                    #### Go through the train_ix and remove the particualr thing; 
                    for i_fold in range(n_folds):
                        
                        train_fold_full = train_ix[i_fold]
                        test_fold  = test_ix[i_fold]

                       #### train the model and predict held-out data ###
                        y_pred_lo[test_fold, :], coef, intc = train_and_pred(spks_tm1_tsk, spks_tm0_tsk, push_tm0_tsk, 
                            train_fold_full, test_fold, alpha_spec, KG)

                    #### Get the r2 of held out data ###
                    r2_alph[ialpha] = util_fcns.get_R2(spks_tm0_tsk, y_pred_lo)

                #### Get the max alpha ###
                print('%s, %d, tsk=%d, '%(animal, day_ix, tsk))
                print(r2_alph)

                ix_max = np.argmax(r2_alph)
                max_alpha[animal, day_ix, tsk] = alphas[ix_max]
                max_alpha[animal, day_ix, tsk, 'N_trn'] = len(train_fold_full)

    #### Save max_alpha ###
    pickle.dump(max_alpha, open(os.path.join(analysis_config.config['grom_pref'], 'tsk_spec_alphas.pkl'), 'wb'))

def subsampPd(dat, ix_keep):
    ############ Re-index everything ###########################
    tmp_dict = {}
    for k in dat.keys():
        tmp_dict[k] = np.array(dat[k][ix_keep])
    return pandas.DataFrame(tmp_dict)

def plot_loo_r2_overall(cat='tsk', mean_sub_tsk_spec = False, zero_alpha = False,
    tsk_spec_alphas = False, yval='r2', nshuffs = 100, n_folds = 5, plot_eig = False,
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'): 
    '''
    For each model type lets plot the held out data vs real data correlations 
    ''' 
    if yval == 'err':
        yfcn = err; 

    elif yval == 'r2':
        yfcn = r2; 

    model_set_number = 6 

    f, ax = plt.subplots(figsize=(3, 3))
    f_spec, ax_spec = plt.subplots(ncols = 2); 
    
    for i_a, animal in enumerate(['grom','jeev']):

        r2_stats = []

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            if plot_eig:
                fmn, axmn = plt.subplots()
                feig, axeig = plt.subplots()

            if mean_sub_tsk_spec:
                ext = '_tsksub'
            else:
                ext = ''

            if zero_alpha:
                ext2 = '_zeroalph'
            elif tsk_spec_alphas:
                ext2 = '_tskalph'
            else:
                ext2 = ''

            if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
                ext3 = ''
            elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0':
                ext3 = '_nocond'

            NLOO_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'], 'loo_ctrl_%s_%s_%d%s%s%s.pkl'%(cat, animal, day_ix, ext, ext2, ext3)), 'rb'))

            ### Load the category dictonary: 
            LOO_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'], 'loo_%s_%s_%d%s%s%s.pkl'%(cat, animal, day_ix, ext, ext2, ext3)), 'rb'))

            KG = util_fcns.get_decoder(animal, day_ix)

            ### Load the true data ###
            spks_true, com_true, mov_true, push_true, com_true_tm1 = get_spks(animal, day_ix)
            if 'analyze_indices' in LOO_dict.keys():
                an_ind = LOO_dict['analyze_indices']
            else:
                an_ind = np.arange(len(spks_true))

            ######## re index #####
            spks_true = spks_true[an_ind, :]
            com_true = com_true[an_ind, :]
            com_true_tm1 = com_true_tm1[an_ind]
            mov_true = mov_true[an_ind]

            ### Load shuffled dynamics --> same as shuffled, not with left out shuffled #####
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)
            pred_spks_shuffle = 10*pred_spks_shuffle[an_ind, :, :]; 

            ### Go through the items that have been held out: 
            left_outters = LOO_dict.keys()
            left_outters = [l for l in left_outters if l != 'analyze_indices']

            lo_true_all = []
            lo_pred_all = []
            nlo_pred_all = []
            shuff_pred_all = []

            r2_stats_spec = []; 
            r2_stats_spec_nlo = [];

            flo, axlo = plt.subplots(ncols=2)
            
            for left_out in left_outters: 

                if len(LOO_dict[left_out].keys()) == 0:
                    pass
                else:
                    #### Get the thing that was left out ###
                    lo_pred = 10*LOO_dict[left_out][-1, -1, -1]
                    lo_ix = LOO_dict[left_out][-1, -1, -1, 'ix']

                    ##### Get predicted spikes ###
                    spks_pred = 10*NLOO_dict[left_out]['y_pred_nlo']
                    #spks_pred = spks_pred - 
                    nneur = spks_pred.shape[1]

                    #### Check that these indices corresond to the left out thing; 
                    check_ix_lo(lo_ix, com_true, com_true_tm1, mov_true, left_out, cat)
                    
                    if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
                        lo_pred_all.append(lo_pred)
                        lo_true_all.append(spks_true[lo_ix, :])
                        nlo_pred_all.append(spks_pred[lo_ix, :])
                        shuff_pred_all.append(pred_spks_shuffle[lo_ix, :, :])

                        #### r2 stats specific ####
                        tmp,_=yfcn(spks_true[lo_ix, :], spks_pred[lo_ix, :])
                        r2_stats_spec_nlo.append([left_out, tmp])

                        tmp2,_=yfcn(spks_true[lo_ix, :], lo_pred)
                        r2_stats_spec.append([left_out, tmp2])
                    
                        ###### PLot R2 by different categories; ######
                        #### What fraction of the data is this left out data point ? #####
                        #axlo[1].plot(left_out, float(len(lo_ix))/float(spks_pred.shape[0]), 'k.')
                        tmp3 = float(len(lo_ix))/float(len(an_ind))
                        print('Len lo_ix %d, len_overall %d = %.1f' %(len(lo_ix), len(an_ind), tmp3))
                        axlo[1].plot(left_out, float(len(lo_ix))/float(len(an_ind)), 'k.')
                        axlo[0].set_title('%s, %d' %(animal, day_ix))
                        axlo[0].plot(left_out, tmp, '.', color=analysis_config.blue_rgb)
                        axlo[0].plot(left_out, tmp2, '.', color='purple')
                        tmp3 = []
                        for i in range(nshuffs):
                            t, _ = yfcn(spks_true[lo_ix, :], pred_spks_shuffle[lo_ix, :, i])
                            tmp3.append(t)
                        util_fcns.draw_plot(left_out, tmp3, 'k', np.array([1.,1.,1.,0.]), axlo[0])
                        axlo[0].set_xlim([-1, np.max(left_outters)+1])

                    elif model_nm == 'hist_1pos_0psh_0spksm_1_spksp_0': 
                        ### want to plot "next action" 
                        ### Here theses are all predictions of tm0 | tm1, NOT conditioned on action 
                        ### Lets see how accurate the action is; 
                        lo_pred_all.append(np.dot(KG, 0.1*lo_pred.T).T)
                        lo_true_all.append(push_true[np.ix_(lo_ix, [3, 5])])
                        nlo_pred_all.append(np.dot(KG, 0.1*spks_pred[lo_ix, :].T).T)
                        shuff_act = []
                        for n in range(nshuffs):
                            shuff_act.append(np.dot(KG, 0.1*pred_spks_shuffle[lo_ix, :, n].T).T)
                        shuff_pred_all.append(np.dstack((shuff_act)))

                    if plot_eig:
                        for i_fold in range(n_folds):
                            try:
                                axmn.plot(np.arange(nneur), LOO_dict[left_out][i_fold, 'intc_lo'], '-', color=analysis_config.pref_colors[left_out], linewidth=.5)
                                axmn.plot(np.arange(nneur), NLOO_dict[left_out][i_fold, 'intc_nlo'], 'k-', linewidth=.5)                    
                            except:
                                pass            
                            hz, decay = plot_pred_fr_diffs.get_ang_td(LOO_dict[left_out][i_fold, 'coef_lo'], plt_evs_gte=.99, dt=0.1)
                            axeig.plot(decay, hz, '.', color=analysis_config.pref_colors[left_out])
                            
                            hz, decay = plot_pred_fr_diffs.get_ang_td(NLOO_dict[left_out][i_fold, 'coef_nlo'], plt_evs_gte=.99, dt=0.1)
                            axeig.plot(decay, hz, 'k.')

            if plot_eig:
                axmn.set_title('%s, day = %d'%(animal, day_ix))
                axeig.set_title('%s, day = %d'%(animal, day_ix))
                fmn.tight_layout()
                feig.tight_layout()

            ##### For this day, plot the R2 comparisons #####
            lo_true_all = np.vstack((lo_true_all))
            lo_pred_all = np.vstack((lo_pred_all))
            nlo_pred_all = np.vstack((nlo_pred_all))
            shuff_pred_all = np.vstack((shuff_pred_all))

            r2_pred_lo, _ = yfcn(lo_true_all, lo_pred_all)
            r2_pred_nlo, _ = yfcn(lo_true_all, nlo_pred_all)

            r2_shuff = []
            for i in range(nshuffs):
                tmp,_ = yfcn(lo_true_all, shuff_pred_all[:, :, i])
                r2_shuff.append(tmp)
            
            r2_stats.append([r2_pred_lo, r2_pred_nlo])
            ax.plot(i_a*10 +day_ix-0.1, r2_pred_nlo, '.', color=analysis_config.blue_rgb, markersize=10)
            ax.plot(i_a*10 +day_ix+0.1, r2_pred_lo, '.', color='purple', markersize=10)
            util_fcns.draw_plot(i_a*10 + day_ix, r2_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
            mn = np.min([r2_pred_nlo, r2_pred_lo, np.mean(r2_shuff)])
            mx = np.max([r2_pred_nlo, r2_pred_lo, np.mean(r2_shuff)])
            ax.plot([i_a*10 + day_ix, i_a*10 + day_ix], [mn, mx], 'k-', linewidth=0.5)

            if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
                ### Animal specific plot 
                r2_stats_spec = np.vstack((r2_stats_spec))
                r2_stats_spec_nlo = np.vstack((r2_stats_spec_nlo))

                ax_spec[i_a].plot(r2_stats_spec[:, 0], r2_stats_spec[:, 1], '.', 
                    color=analysis_config.pref_colors[day_ix])

                ax_spec[i_a].plot(r2_stats_spec_nlo[:, 0]+.2, r2_stats_spec_nlo[:, 1], '^', 
                    color=analysis_config.pref_colors[day_ix])

        ### Vstack r2_stats ####
        if model_nm == 'hist_1pos_0psh_2spksm_1_spksp_0':
            r2_stats = np.vstack((r2_stats))
#           assert(r2_stats.shape[0] == len(range(analysis_config.data_params['%s_ndays'%animal])))
            assert(r2_stats.shape[1] == 2)

            r2_stats = np.mean(r2_stats, axis=0)
            assert(len(r2_stats) == 2)
            # ax.bar(3*i_a, r2_stats[0], width=0.8, color='k', alpha=.2)
            # ax.bar((3*i_a)+1, r2_stats[1], width=0.8, color='k', alpha=.2)

            #### Plot task-specific ####
            ax_spec[i_a].set_ylabel(yval)
            ax_spec[i_a].set_xlabel('Diff %s s'%(cat))


    ax.set_title('Cat: %s' %cat)
    ax.set_ylabel(yval)
    ax.set_xlim([-1, 14])
    ax.set_xticks([])
    #ax.set_xticks([0, 1, 3, 4])
    #ax.set_xticklabels(['Dyn\nG', 'H-O Dyn\nG', 'Dyn\nJ', 'H-O Dyn\nJ'], rotation=45)
    f.tight_layout()
    f_spec.tight_layout()

    util_fcns.savefig(f, 'held_out_cat%s_%s'%(cat, model_nm))

######## Plot whether the move-speicfic command activity has the right structure #####
def plot_pop_dist_corr_COMMAND(nshuffs=10, min_commands = 15): 

    cat = 'com'
    model_set_number = 6
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'    

    f, ax = plt.subplots(figsize=(2, 3))
    f_err, ax_err = plt.subplots(figsize=(2, 3))
    
    for i_a, animal in enumerate(['grom', 'jeev']):

        model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
        model_dict = pickle.load(open(model_fname, 'rb'))

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load the category dictonary: 
            LOO_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'], 'loo_%s_%s_%d.pkl'%(cat, animal, day_ix)), 'rb'))

            ### Load the true data ###
            spks_true, com_true, mov_true, push_true, _ = get_spks(animal, day_ix)

            ### Load predicted daata freom the model 
            spks_pred = 10*model_dict[day_ix, model_nm]

            ### Load shuffled dynamics --> same as shuffled, not with left out shuffled #####
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)

            pred_spks_shuffle = 10*pred_spks_shuffle; 
            
            ### Go through the items that have been held out: 
            left_outters = LOO_dict.keys()

            pop_dist_true = []
            pop_dist_pred = []
            pop_dist_pred_lo = []
            pop_dist_shuff = {}
            for i in range(nshuffs):
                pop_dist_shuff[i] = []

            ### For each command #####
            for left_out in left_outters:        

                if len(LOO_dict[left_out].keys()) == 0: 
                    pass 
                else: 
                    mag_ang_mov = LOO_dict[left_out].keys()

                    #### Get the movements that correspond to this left otu command
                    #### Ignore the keys with "ix" and the keys that are [-1, -1, -1]
                    mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[0]*8 + m[1] == left_out])
                    mag_ang_mov_ix = []
                    for m in mag_ang_mov:
                        
                        m2 = list(m)
                        m2.append('ix')
                        mag_ang_mov_ix.append(m2) 

                    for i_m, mam in enumerate(mag_ang_mov):
                        mn_est = 10*np.mean(LOO_dict[left_out][tuple(mam)], axis=0)
                        mam_ix = LOO_dict[left_out][tuple(mag_ang_mov_ix[i_m])]

                        for i_m2, mam2 in enumerate(mag_ang_mov[i_m+1:]):
                            mn_est2 = 10*np.mean(LOO_dict[left_out][tuple(mam2)], axis=0)
                            mam_ix2 = LOO_dict[left_out][tuple(mag_ang_mov_ix[i_m2+i_m+1])]

                            #### Match indices 
                            ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_true[np.ix_(mam_ix, [3, 5])], 
                                push_true[np.ix_(mam_ix2, [3, 5])], psig=.05)

                            if len(ix1_1) >= min_commands and len(ix2_2) >= min_commands:

                                mn_true = np.mean(spks_true[mam_ix[ix1_1], :], axis=0)
                                mn_pred = np.mean(spks_pred[mam_ix[ix1_1], :], axis=0)
                                mn_shuff = np.mean(pred_spks_shuffle[mam_ix[ix1_1], :, :], axis=0)

                                mn_true2 = np.mean(spks_true[mam_ix2[ix2_2], :], axis=0)
                                mn_pred2 = np.mean(spks_pred[mam_ix2[ix2_2], :], axis=0)
                                mn_shuff2 = np.mean(pred_spks_shuffle[mam_ix2[ix2_2], :, :], axis=0)

                                #### Save the differences #####
                                for i in range(nshuffs):
                                    pop_dist_shuff[i].append(plot_pred_fr_diffs.nplanm(mn_shuff[:, i], mn_shuff2[:, i]))

                                pop_dist_true.append(plot_pred_fr_diffs.nplanm(mn_true, mn_true2))
                                pop_dist_pred.append(plot_pred_fr_diffs.nplanm(mn_pred, mn_pred2))
                                pop_dist_pred_lo.append(plot_pred_fr_diffs.nplanm(mn_est, mn_est2))

            print('n comparisons %d'%(len(pop_dist_true)))
            plot_LO_means(pop_dist_true, pop_dist_pred, pop_dist_pred_lo, pop_dist_shuff, ax, ax_err, nshuffs, i_a*10 + day_ix,
                animal, day_ix, title_str='command')
    
    save_ax_cc_err(ax, ax_err, f, f_err, 'com')

def plot_pop_dist_corr_MOV_TSK(nshuffs=2, cat='mov', min_commands=15):

    model_set_number = 6
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'    

    f, ax = plt.subplots(figsize=(2, 3))
    f_err, ax_err = plt.subplots(figsize=(2, 3))
    
    for i_a, animal in enumerate(['grom', 'jeev']):

        model_fname = analysis_config.config[animal+'_pref']+'tuning_models_'+animal+'_model_set'+str(model_set_number)+'_.pkl'
        model_dict = pickle.load(open(model_fname, 'rb'))

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):

            ### Load the category dictonary: 
            LOO_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'], 'loo_%s_%s_%d.pkl'%(cat, animal, day_ix)), 'rb'))

            ### Load the true data ###
            spks_true, com_true, mov_true, push_true, _ = get_spks(animal, day_ix)

            ### Load predicted daata freom the model 
            spks_pred = 10*model_dict[day_ix, model_nm]

            ### Load shuffled dynamics --> same as shuffled, not with left out shuffled #####
            pred_spks_shuffle = plot_generated_models.get_shuffled_data_v2(animal, day_ix, model_nm, nshuffs = nshuffs, 
                testing_mode = False)

            pred_spks_shuffle = 10*pred_spks_shuffle; 
            
            ### Go through the items that have been held out: 
            left_outters = LOO_dict.keys()

            true_diff = []
            pred_diff = []
            pred_lo_diff = []
            already_assessed = []
            shuff_diff = {}
            
            for i in range(nshuffs): shuff_diff[i] = []

            for left_out in left_outters:

                if len(LOO_dict[left_out].keys()) == 0: 
                    pass 
                else: 
                
                    mag_ang_mov = LOO_dict[left_out].keys()

                    #### get the commands associated with this movement ####
                    if cat == 'mov':
                        mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[2] == left_out])
                    
                    elif cat == 'com_ang':
                        mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[1] == left_out])
                    
                    elif cat == 'com_mag':
                        mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[0] == left_out and left_out <4])

                    elif cat == 'tsk':
                        if left_out == 0:
                            mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[2] < 10 and m[2] >= 0.])
                        
                        elif left_out == 1:
                            mag_ang_mov = np.array([m for m in mag_ang_mov if len(m) == 3 and m[2] >= 10])

                    
                    mag_ang_mov_ix = []
                    for m in mag_ang_mov:
                        m2 = list(m)
                        m2.append('ix')
                        mag_ang_mov_ix.append(m2) 

                    ######## This list has common movements / tasks ########
                    for i_m, mam in enumerate(mag_ang_mov):

                        pred_est_LO = 10*np.mean(LOO_dict[left_out][tuple(mam)], axis=0)
                        mam_ix = LOO_dict[left_out][tuple(mag_ang_mov_ix[i_m])]

                        ###### Get command ang/mag/mov
                        com_ang, com_mag, mov = mam;

                        ##### Ix sort ######
                        ix_sort = np.nonzero((com_true[:, 0] == com_ang) & (com_true[:, 1] == com_mag))[0]

                        movements_unique = np.unique(mov_true[ix_sort])
                        movements = mov_true[ix_sort]

                        ##### Go through the other movements #####
                        for i_m2, mov2 in enumerate(movements_unique): 

                            ##### Make sure not comparing the smae movements 
                            if mov != mov2 and tuple((com_ang, com_mag, mov, mov2)) not in already_assessed and tuple((com_ang, com_mag, mov2, mov)) not in already_assessed:

                                ix2 = np.nonzero(movements == mov2)[0]
                                ix2_all = ix_sort[ix2]

                                if len(ix2) >= min_commands:
                                    
                                    #### Match indices 
                                    ix1_1, ix2_2, niter = plot_fr_diffs.distribution_match_mov_pairwise(push_true[np.ix_(mam_ix, [3, 5])], 
                                        push_true[np.ix_(ix2_all, [3, 5])], psig=.05)

                                    if len(ix1_1) >= min_commands and len(ix2_2) >= min_commands:

                                        already_assessed.append(tuple((com_ang, com_mag, mov, mov2)))

                                        tru_est = np.mean(spks_true[mam_ix[ix1_1], :], axis=0)
                                        pred_est = np.mean(spks_pred[mam_ix[ix1_1], :], axis=0)
                                        shuff_est = np.mean(pred_spks_shuffle[mam_ix[ix1_1], :, :], axis=0)

                                        tru_est2 = np.mean(spks_true[ix2_all[ix2_2], :], axis=0)
                                        pred_est2 = np.mean(spks_pred[ix2_all[ix2_2], :], axis=0)
                                        shuff_est2 = np.mean(pred_spks_shuffle[ix2_all[ix2_2], :], axis=0)


                                        true_diff.append(plot_pred_fr_diffs.nplanm(tru_est, tru_est2))
                                        pred_diff.append(plot_pred_fr_diffs.nplanm(pred_est, pred_est2))

                                        ### Compare this to predicted ###
                                        pred_lo_diff.append(plot_pred_fr_diffs.nplanm(pred_est_LO, pred_est2))
                                        for i in range(nshuffs):
                                            shuff_diff[i].append(plot_pred_fr_diffs.nplanm(shuff_est[:, i], shuff_est2[:, i]))
                
            print("num diffs %d" %(len(true_diff)))
            
            #### For each movement, comparison b/w LO commadn estimate for a given movement vs. predicted 
            plot_LO_means(true_diff, pred_diff, pred_lo_diff, shuff_diff, ax, ax_err, nshuffs, i_a*10 + day_ix,
                animal, day_ix, title_str=cat)
    
    save_ax_cc_err(ax, ax_err, f, f_err, cat)

def plot_LO_means(pop_dist_true, pop_dist_pred, pop_dist_pred_lo, pop_dist_shuff, ax, ax_err, nshuffs, xpos,
    animal, day_ix, title_str=''):

    ###### For each day / animal plot the CC and Error? ####
    #####Plot all the pts too ####
    ### R2 and Error ###

    f_spec, ax_spec = plt.subplots(figsize=(3, 3)); 
    ax_spec.plot(pop_dist_true, pop_dist_pred, 'k.', markersize=5.)
    ax_spec.plot(pop_dist_true, pop_dist_pred_lo, '.', color='purple', markersize=5.) 
    
    ax_spec.set_title('%s: %d, CC lo =%.3f, CC pred =%.3f' %(animal, day_ix,
        cc(pop_dist_true, pop_dist_pred_lo), cc(pop_dist_true, pop_dist_pred)), fontsize=10)
    f_spec.tight_layout()
    if animal == 'grom' and day_ix == 0:
        util_fcns.savefig(f_spec, 'eg_leave_one_out_command_%s_%d_%s'%(animal, day_ix, title_str))

    #### Get correlation and plot 
    c1 = cc(pop_dist_true, pop_dist_pred)
    c2 = cc(pop_dist_true, pop_dist_pred_lo)
    
    ax.plot(xpos-.1, c1, '.', color=analysis_config.blue_rgb, markersize=10)
    ax.plot(xpos+.1, c2, '.', color='purple', markersize=10)
    
    rv_shuff = []
    for i in range(nshuffs):
        rv_shuff.append(cc(pop_dist_true, pop_dist_shuff[i]))
    util_fcns.draw_plot(xpos, rv_shuff, 'k', np.array([1., 1., 1., 0.]), ax)
    ax.plot([xpos, xpos], [np.mean(rv_shuff), np.max(np.array([c1, c2]))], 'k-', linewidth=0.5)


    ################ Get error ################ 
    e1 = plot_pred_fr_diffs.mnerr(pop_dist_true, pop_dist_pred)
    e2 = plot_pred_fr_diffs.mnerr(pop_dist_true, pop_dist_pred_lo)
    ax_err.plot(xpos-.1, e1, '.', color=analysis_config.blue_rgb, markersize=10)
    ax_err.plot(xpos+.1, e2, '.', color='purple', markersize=10)
    
    er_shuff = []
    for i in range(nshuffs):
        er_shuff.append(plot_pred_fr_diffs.mnerr(pop_dist_true, pop_dist_shuff[i]))
    util_fcns.draw_plot(xpos, er_shuff, 'k', np.array([1., 1., 1., 0.]), ax_err)
    ax_err.plot([xpos, xpos], [np.mean(er_shuff), np.max(np.array([e1, e2]))], 'k-', linewidth=0.5)

######## Test whether activity is move specific or not ##############
def move_spec_dyn(n_folds=5, zero_alpha = False):
    ''' 
    method to test whether movements can be better explained by 
    mov-specific or general dynamics 

    try to match the training data sizes 
    ''' 
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0'
    model_set_number = 6 
    ridge_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'] , 'max_alphas_ridge_model_set%d.pkl'%model_set_number), 'rb')); 

    for i_a, animal in enumerate(['grom', 'jeev']):

        input_type = analysis_config.data_params['%s_input_type'%animal]
        ord_input_type = analysis_config.data_params['%s_ordered_input_type'%animal]

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            print('Starting %s, Day %d' %(animal, day_ix))
            f, ax = plt.subplots(figsize = (8, 8))

            #### Get ridge alpha ####
            if zero_alpha:
                alpha_spec = 1.
            else:
                alpha_spec = ridge_dict[animal][0][day_ix, model_nm]
            KG = util_fcns.get_decoder(animal, day_ix)

            #### Get data ####
            data, data_temp, sub_spikes, sub_spk_temp_all, sub_push_all = generate_models_utils.get_spike_kinematics(animal, input_type[day_ix], 
            ord_input_type[day_ix], 1, within_bin_shuffle = False, day_ix = day_ix, skip_plot = True)

            ############## Data checking ##############
            nneur = sub_spikes.shape[1]
            for n in range(nneur):
                assert(np.allclose(sub_spk_temp_all[:, 0, n], data_temp['spk_tm1_n%d'%n]))
                assert(np.allclose(sub_spk_temp_all[:, 1, n], data_temp['spk_tm0_n%d'%n]))

            ############## Data checking ##############
            push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T

            #### Add teh movement category ###
            data_temp['mov'] = data_temp['trg'] + 10*data_temp['tsk']
            data_temp['com'], _, _ = get_com(sub_push_all, animal, day_ix)
            
            ############## Get subspikes ##############
            spks_tm1 = sub_spk_temp_all[:, 0, :]
            spks_tm0 = sub_spk_temp_all[:, 1, :]
            push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T

            #### Add teh movement category ###
            data_temp['mov'] = data_temp['trg'] + 10*data_temp['tsk']

            ######### Use these same indices ---> 
            _, Agen, _ = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                    np.arange(len(spks_tm1)), np.array([0]), alpha_spec, KG)

            #### Make 5 folds --> all must exclude the thing you want to exclude, but distribute over the test set; 
            test_ix_mov, train_ix_mov, _ = generate_models_utils.get_training_testings_condition_spec(n_folds, data_temp)
            _, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)

            for i_mov, mov in enumerate(np.unique(data_temp['mov'])):

                y_g = []; y_gf = []
                y_s = []; y_sb = []
                tru = []

                for i_f in range(n_folds):
    
                    ######### Use these same indices ---> 
                    ypred_mov, _, _ = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_ix_mov[i_f, mov], test_ix_mov[i_f, mov], alpha_spec, KG)

                    ypred_mov_b, _, b_spec = train_spec_b(spks_tm1, spks_tm0, push_tm0, 
                            train_ix_mov[i_f, mov], test_ix_mov[i_f, mov], KG, Agen)

                    N_trn_mov = len(train_ix_mov[i_f, mov])
                    N_trl_all = len(train_ix[i_f])
                    ix_sub = np.random.permutation(N_trl_all)[:N_trn_mov]
                    train_gen = train_ix[i_f][ix_sub]

                    ######### Use these same indices for testing, 
                    ypred_gen, _, _ = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_gen, test_ix_mov[i_f, mov], alpha_spec, KG)

                    train_full = np.array([i for i in train_ix[i_f] if i not in test_ix_mov[i_f, mov]])
                    ypred_gen_full_TD, _, _ = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_full, test_ix_mov[i_f, mov], alpha_spec, KG)

                    ######## True spks; 
                    true_sub = spks_tm0[test_ix_mov[i_f, mov], :]

                    y_g.append(ypred_gen)
                    y_gf.append(ypred_gen_full_TD)
                    y_s.append(ypred_mov)
                    y_sb.append(ypred_mov_b)
                    tru.append(true_sub)

                ####### get r2 
                r2_spec = util_fcns.get_R2(np.vstack((tru)), np.vstack((y_s)))
                r2_bspec_Agen = util_fcns.get_R2(np.vstack((tru)), np.vstack((y_sb)))
                r2_gen  = util_fcns.get_R2(np.vstack((tru)), np.vstack((y_g)))
                r2_gen_f= util_fcns.get_R2(np.vstack((tru)), np.vstack((y_gf)))

                if i_mov == 0:
                    #ax.plot(mov, r2_gen, 'k.', markersize=15, label='Gen dyn fit w $N_{mov}$') #### General dynamnics, trainign dataset size matched to mov spec; 
                    ax.plot(mov, r2_gen_f, 'k^', label='Gen dyn fit w $N_{full}$') ### General dynamics, full training data set size; 
                    ax.plot(mov, r2_bspec_Agen, 'd', color=util_fcns.get_color(mov), markersize=10,
                        label='$A_{gen}, b_{mov}$', markeredgewidth=3.) ### move specific b, general A 
                    #ax.plot(mov, r2_spec, '.', color=util_fcns.get_color(mov), markersize=15,
                    #    label='Mov spec dyn fit w $N_{mov}$', markeredgewidth=3.) ### move specific predictions 
                else:
                    #ax.plot(mov, r2_gen, 'k.', markersize=15) #### General dynamnics, trainign dataset size matched to mov spec; 
                    ax.plot(mov, r2_gen_f, 'k^') ### General dynamics, full training data set size; 
                    ax.plot(mov, r2_bspec_Agen, 'd', color=util_fcns.get_color(mov), markersize=10,
                        markeredgewidth=3.) ### move specific b, general A 
                    #ax.plot(mov, r2_spec, '.', color=util_fcns.get_color(mov), markersize=15,
                    #    markeredgewidth=3.) ### move specific predictions                     
                
                #ax.plot([mov, mov, mov, mov], [r2_gen, r2_gen_f, r2_bspec_Agen, r2_spec], 'k-', linewidth=.5)
            
            ax.set_xlabel('Movement')
            ax.set_ylabel('R2')
            ax.legend()
            ax.set_title('%s, %d' %(animal, day_ix))
            f.tight_layout()

