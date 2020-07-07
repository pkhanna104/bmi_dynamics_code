import numpy as np
import analysis_config
import feedback_controllers
from lqr_simulation import Combined_Curs_Brain_LQR_Simulation, Experiment_Cursor
import obstacle_fcns
import scipy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pickle
import tables

import os, pickle
import analysis_config

class NHPBrain_RidgeOffsDyn(object):
    '''
    Using estimates of neural dynamics from data; 
    '''

    def __init__(self, day, animal, shuffle):

        A, offs = get_saved_RidgeOffs(day = day, animal = animal, shuffle = shuffle)

        ##### Append offset to A; 
        nNeur = A.shape[0]

        ##### A.shape = (n+1) x (n+1)
        A = np.hstack((A, offset[:, np.newaxis]))
        A = np.vstack((A, np.hstack(( np.zeros((nNeur, )), [1] ))))
        self.A = np.mat(A)

        #### Inputs to each; (n+1) x (n) #######
        B = np.eye(nNeur)
        B = np.vstack((B, np.zeros((nNeur)), ))
        self.B = np.mat(B)

        ##### Estimated neural observations #####
        self.nobs = nNeur + 1
        self.nNeur = nNeur
        self.obs = np.mat(np.hstack(( np.zeros((nNeur)), [1] ))[:, np.newaxis])

    def get_next_state(self, input1):
        assert(input1.shape[0] == self.nNeur)
        inp = np.mat(input1).reshape(-1, 1)

        next = np.dot(self.A, self.obs) + np.dot(self.B, inp)
        self.obs = next.copy()

class ComboStateRidgeOffs(object):
    def __init__(self, cursor, brain):

        nCurs = len(cursor.keep_offset)

        #### Make a Kalman gain account for neural offset;  
        KG_offs = np.hstack((cursor.B, np.zeros((nCurs, 1))))


        self.A = np.block([[ brain.A, np.zeros((brain.nobs, nCurs))],
                           [ np.dot(KG_offs, brain.A), cursor.A]])

        self.B = np.block([[brain.B], [np.dot(KG_offs, brain.B)]])

        self.state = np.vstack(( brain.obs, cursor.state))
        self.brain_state_ix = np.arange(brain.nobs)
        self.cursor_state_ix = np.arange(brain.nobs, brain.nobs + cursor.nstates)

        self.cursor = cursor; 
        self.brain = brain; 

class Combined_Curs_Brain_LQR_Data_ModelRidgeOffs(Combined_Curs_Brain_LQR_Simulation):

    def __init__(self, day, shuffle, R = 10000, task = 'co'):

        ####### Get cursor ########
        keep_offset = True
        self.curs = Experiment_Cursor(day, keep_offset)

        ####### Get brain ######
        self.brain = NHPBrain_RidgeOffsDyn(day, 'grom', shuffle)

        ###### Get a combined state #####

######### Get saved ridge regression ########
def get_saved_RidgeOffs(day=0, animal='grom', shuffle = False):
    
    #### Try loading from saved place ####
    if os.path.exists(config['lqr_sim_saved_dyn'] + '%s_%d.pkl' %(animal, day)):
        dat = pickle.load(open(config['lqr_sim_saved_dyn'] + '%s_%d.pkl' %(animal, day), 'rb'))
        model = dat['model']

    else:
        ##### Load data #####
        dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
        dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(7), 'rb'))
        
        i_m = 2; #### General dynamics; 
        n_folds = 5; 
        i_f = 0; 
        
        model = dat[day, dyn_model, n_folds*i_m + i_f, i_m, 'model']
        pickle.dump(dict(model=model), open(config['lqr_sim_saved_dyn'] + '%s_%d.pkl' %(animal, day), 'wb'))

    A = np.mat(model.coef_)
    offs = np.mat(model.intercept_[:, np.newaxis])
    return A, offs

