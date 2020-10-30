import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import os 

from online_analysis import util_fcns, generate_models, generate_models_list, generate_models_utils
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
        tm0, _ = generate_models.get_temp_spks_ix(dat['Data'])

        ### Get subsampled 
        self.spks = spks0[tm0, :]
        self.push = push0[tm0, :]
        self.command_bins = util_fcns.commands2bins([self.push], mag_boundaries, self.animal, self.day_ix, 
                                       vel_ix=[3, 5])[0]
        self.move = move0[tm0]

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

#### model fitting utls ###########
def train_and_pred(spks_tm1, spks, push, train, test, alpha, KG):
    """Summary
    
    Parameters
    ----------
    data_temp : TYPE
        dictonary from get_spike_kinematics 
    train : TYPE
        np.array of training indicies 
    test : TYPE
        np.array of test indices 
    alpha : TYPE
        ridge parameter for regression for this day from swept alphas. 
    nneur : int 
        number of neurons
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
    

    ### Row is a variable, column is an observation for np.cov
    W = np.cov((y_train - y_train_est).T)
    assert(W.shape[0] == X_train.shape[1])

    ### Now estimate held out data with conditioning ###
    cov12 = np.dot(KG, W).T
    cov21 = np.dot(KG, W)
    cov22 = np.dot(KG, np.dot(W, KG.T))
    cov22I = np.linalg.inv(cov22)

    T = len(test)

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

    return np.vstack((pred_w_cond))

def get_com(tby2_push, animal, day_ix):
    command_bins = util_fcns.commands2bins([tby2_push], mag_boundaries, animal, day_ix, 
                                       vel_ix=[0, 1])[0]
    return command_bins[:, 0]*8 + command_bins[:, 1]

######## Bar plots by task / target / command ####
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

######## Fit general model ######
def fit_predict_loo_model(cat='tsk', n_folds = 5, min_num_per_cat_lo = 15,
    model_nm = 'hist_1pos_0psh_2spksm_1_spksp_0', model_set_number = 6):
    """Summary
    
    Parameters
    ----------
    cat : str, optional
        Description
    """
    cat_dict = {}
    cat_dict['tsk'] = ['tsk', np.arange(2)]
    cat_dict['mov'] = ['mov', np.unique(np.hstack((np.arange(8), np.arange(10, 20), np.arange(10, 20)+.1)))]
    cat_dict['com'] = ['com', np.arange(32)]

    #### which category are we dealing with ####
    leave_out_field = cat_dict[cat][0]
    leave_out_cats = cat_dict[cat][1]

    ### get the right alphase for the model ####

    ridge_dict = pickle.load(open(os.path.join(analysis_config.config['grom_pref'] , 'max_alphas_ridge_model_set%d.pkl'%model_set_number), 'rb')); 

    for i_a, animal in enumerate(['grom', 'jeev']):

        input_type = analysis_config.data_params['%s_input_type'%animal]
        ord_input_type = analysis_config.data_params['%s_ordered_input_type'%animal]

        for day_ix in range(analysis_config.data_params['%s_ndays'%animal]):
            
            print('Starting %s, Day %d' %(animal, day_ix))

            #### Get ridge alpha ####
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
            
            ############## Get subspikes ##############
            spks_tm1 = sub_spk_temp_all[:, 0, :]
            spks_tm0 = sub_spk_temp_all[:, 1, :]
            
            push_tm0 = np.vstack((data_temp['pshx_tm0'], data_temp['pshy_tm0'])).T

            #### Add teh movement category ###
            data_temp['mov'] = data_temp['trg'] + 10*data_temp['tsk']
            data_temp['com'] = get_com(sub_push_all, animal, day_ix)
            
            #### Make 5 folds --> all must exclude the thing you want to exclude, but distribute over the test set; 
            test_ix, train_ix = generate_models_utils.get_training_testings(n_folds, data_temp)

            #### setup data storage 
            LOO_dict = {}

            for lo in leave_out_cats: 
                LOO_dict[lo] = {}

                y_pred = np.zeros_like(spks_tm0) + np.nan
    
                #### Which indices must be removed from all training sets? 
                ix_rm = np.nonzero(data_temp[leave_out_field] == lo)[0]

                if len(ix_rm) >= min_num_per_cat_lo: 

                    #### Go through the train_ix and remove the particualr thing; 
                    for i_fold in range(5):
                        print('Starting fold %d for LO %1.f' %(i_fold, lo))

                        train_fold_full = train_ix[i_fold]
                        test_fold  = test_ix[i_fold]

                        #### removes indices 
                        train_fold = np.array([ i for i in train_fold_full if i not in ix_rm])

                        #### train the model and predict held-out data ###
                        y_pred[test_fold, :] = train_and_pred(spks_tm1, spks_tm0, push_tm0, 
                            train_fold, test_fold, alpha_spec, KG)
                        
                    
                    #### Now go through and compute movement-specific commands ###
                    assert(np.sum(np.isnan(y_pred)) == 0)

                    for mag in range(4):
                        for ang in range(8):
                            for mov in np.unique(data_temp['mov']):
                                mc = np.where((data_temp['com'] == mag*8 + ang) & (data_temp['mov'] == mov))
                                assert(type(mc) is tuple)
                                mc = mc[0]
                                
                                if len(mc) >= 15:
                                    LOO_dict[lo][mag, ang, mov] = y_pred[mc, :].copy()
                                    LOO_dict[lo][mag, ang, mov, 'ix'] = mc.copy()

                #### Save this LOO ####
            pickle.dump(LOO_dict, open(os.path.join(analysis_config.config['grom_pref'], 'loo_%s_%s_%d.pkl'%(cat, animal, day_ix)), 'wb'))

def plot_loo_model(): 
    '''
    For each model type lets plot the held out data vs real data correlations 
        -- Also the R2? 
        -- Another way to 
    ''' 


