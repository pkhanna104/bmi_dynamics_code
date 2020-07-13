#### Config ####
import analysis_config

#### Help code ###
import feedback_controllers
from lqr_simulation import Combined_Curs_Brain_LQR_Simulation, Experiment_Cursor, Cursor
from online_analysis import plot_generated_models
import obstacle_fcns

#### Pkgs ##
import numpy as np
import scipy
import pickle
import tables
import os
import matplotlib.pyplot as plt

class NHPBrain_RidgeOffsDyn(object):
    '''
    Using estimates of neural dynamics from data; 
    Assuming inputs is full dimensional (# neurons)
    Assume "B" is identity --> each input projects to eaach neural dimension 
    '''

    def __init__(self, day, animal, shuffle, with_intercept = True, 
        state_noise = 0, zeroA = False, zerointc = False):

        if with_intercept:
            A, offset, mu, cov = get_saved_RidgeOffs(day = day, animal = animal, shuffle = shuffle,
                with_intercept = with_intercept, zeroA = zeroA, zerointc = zerointc)
        else:
            A, mu, cov = get_saved_RidgeOffs(day = day, animal = animal, shuffle = shuffle,
                with_intercept = with_intercept, zeroA = zeroA, zerointc = zerointc)
        
        ##### Append offset to A; 
        nNeur = A.shape[0]

        if with_intercept:
            ##### A.shape = (n+1) x (n+1)
            A = np.hstack((A, offset))
            A = np.vstack((A, np.hstack(( np.zeros((nNeur, )), [1] ))))

        self.A = np.mat(A)

        #### Inputs to each; (n+1) x (n) #######
        B = np.eye(nNeur)
    
        if with_intercept:
            B = np.vstack((B, np.zeros((nNeur)), ))
    
        self.B = np.mat(B)

        ##### Estimated neural observations #####
        if with_intercept:
            self.nobs = nNeur + 1
            self.nstates = nNeur + 1 #### using this for backward compatibility
            self.obs = np.mat(np.hstack(( np.zeros((nNeur)), [1] ))[:, np.newaxis])
        else:
            self.nobs = nNeur
            self.nstates = nNeur  #### using this for backward compatibility
            self.obs = np.mat(np.hstack(( np.zeros((nNeur)) ))[:, np.newaxis])
        

        self.W = np.eye(self.nstates)*state_noise
        if with_intercept:
            self.W[-1, -1] = 0

        self.nNeur = nNeur
        self.ninputs = nNeur

        self.neural_mu = mu; 
        self.neural_cov = cov; 
        self.with_intercept = with_intercept

    def get_next_state(self, input1):
        
        assert(input1.shape[0] == self.nNeur)
        inp = np.mat(input1).reshape(-1, 1)

        next = np.dot(self.A, self.obs) + np.dot(self.B, inp)
        next += np.random.multivariate_normal(np.zeros((self.nstates)), self.W)
        self.obs = next.copy()

    def get_reset_state(self):
        return np.mat(np.hstack(( self.brain_target ))[:, np.newaxis])

class SimBrain_RidgeOffsDyn(object):
    '''
    simulation of dynamics -- for understanding
    Assume inputs is full dimensional (#neurons)
    '''
    def __init__(self, nNeur, eig_decay, freq, offset):
        
        # Assign:
        self.with_intercept = True
        self.nobs = self.nstates = nNeur + 1
        self.nNeur = nNeur
        self.ninputs = nNeur
        self.obs = np.mat(np.hstack(( np.zeros((nNeur)), [1] ))[:, np.newaxis])

        # Get A: 
        self.A = np.mat(self.create_A(nNeur, eig_decay, freq, offset))

        assert(self.A.shape[0] == self.A.shape[1] == self.nstates)

        # Get B: 
        B = np.eye(nNeur)
        self.B = np.mat(np.vstack((B, np.zeros((nNeur)) )))

        assert(self.B.shape[0] == self.A.shape[1] == self.nstates)

        # Get W 
        self.W = np.eye(self.nstates)*0.#state_noise
        if self.with_intercept:
            self.W[-1, -1] = 0


    def create_A(self, nstates, decay, ang_frequency, offset):
        ### ang_frequency --> in degrees
        ### Keep the starting A as eye: 
        if np.mod(nstates, 2) != 0:
            raise Exception('nstates in sim must be even! ')
        As = []
        for i in range(nstates/2):
            omega = ang_frequency * np.pi / 180. + 0.0001*np.random.randn()

            A1 = (1-.1*i)*decay*np.array([[np.cos(omega), -1*np.sin(omega)],
                  [np.sin(omega), np.cos(omega)]])
            As.append(A1)

        A = scipy.linalg.block_diag(*As)
        A = np.vstack((np.hstack((A, offset[:, np.newaxis])), np.zeros((len(A)+1))))
        A[-1, -1] = 1.
        return A

    def get_reset_state(self):
        return np.mat(np.hstack(( np.zeros((self.nNeur)), [1] ))[:, np.newaxis])

class ComboStateRidgeOffs(object):
    def __init__(self, cursor, brain):

        #### Make a Kalman gain account for neural offset;  
        if brain.with_intercept:
            KG_offs = np.hstack((cursor.B, np.zeros((cursor.nstates, 1))))
        else:
            KG_offs = cursor.B

        self.A = np.block([[ brain.A, np.zeros((brain.nobs, cursor.nstates))],
                           [ np.dot(KG_offs, brain.A), cursor.A]])

        self.B = np.block([[brain.B], [np.dot(KG_offs, brain.B)]])

        self.state = np.vstack(( brain.obs, cursor.state))
        self.brain_state_ix = np.arange(brain.nobs)
        self.cursor_state_ix = np.arange(brain.nobs, brain.nobs + cursor.nstates)

        self.cursor = cursor; 
        self.brain = brain; 
        self.nstates = self.A.shape[0]
        self.W = np.eye(self.nstates)

        for i_n in range(self.brain.nstates):
            self.W[i_n, i_n] = self.brain.W[i_n, i_n]
        
        for i_n in range(self.brain.nstates, self.nstates):
            self.W[i_n, i_n] = 0.

    def get_curs_reset_state(self):
        if self.cursor.keep_offset:
            return np.mat([0., 0., 0., 0., 1.]).reshape(-1,1)
        else:
            return np.mat([0., 0., 0., 0.]).reshape(-1, 1)
    def get_next_state(self, input1):
        
        inp = np.mat(input1)
        assert(len(inp) == self.brain.ninputs)

        ############## noiseless for now ##############
        ## This is noise in the neural state: 
        #noise = np.random.multivariate_normal(np.zeros((len(self.W))), self.W)
        #noise = np.mat(noise).reshape(-1, 1)

        # if self.cursor.input_type == 'state':
        #     noise = np.vstack(( noise, np.dot(self.cursor.B, noise) ))

        # elif self.cursor.input_type == 'obs':
        #     noise = np.vstack(( noise, np.dot(self.cursor.B, np.dot(self.brain.C, noise))))
        
        next = np.dot(self.A, self.state) + np.dot(self.B, inp) #+ noise
        next = next + np.mat(np.random.multivariate_normal(np.zeros((self.nstates)), self.W)).T
        
        ### Update the input: 
        self.state = next.copy()
        
    def reset_state(self):
        brain_reset = self.brain.get_reset_state()
        curs_reset = self.get_curs_reset_state()
        self.state = np.vstack((brain_reset, curs_reset))

class Combined_Curs_SimBrain_LQR_Data_ModelRidgeOffs(Combined_Curs_Brain_LQR_Simulation):

    def __init__(self, R = 10000, 
        eig_decay = 0.99, dyn_freq = 0., task = 'co'):
        
        # 10 states :
        #self.curs = Cursor(nNeur, True)
        self.curs = Cursor(2, keep_offset = True)
        self.keep_offset = True

        # 10 states, 10 inputs, no noise, reasonable dynamics
        offset = 1+np.zeros((2, ))
        self.brain = SimBrain_RidgeOffsDyn(2, eig_decay, dyn_freq, offset)

        ### Compute the fixed point and make this the brain target 
        #ImA = np.linalg.inv(np.eye(self.brain.nNeur) - self.brain.A[:-1, :-1])
        #fix_pt = np.dot(ImA, self.brain.A[:-1, -1])
        #self.brain_target = np.hstack(( np.squeeze(np.array(fix_pt)), [1] ))
        self.brain.brain_target = None; 

        # Combined states: 
        self.combostate = ComboStateRidgeOffs(self.curs, self.brain)

        self.setup_lqr(task, R)

class Combined_Curs_Brain_LQR_Data_ModelRidgeOffs(Combined_Curs_Brain_LQR_Simulation):

    def __init__(self, day, shuffle, R = 10000, task = 'co', with_intercept = False,
        state_noise = 0., zeroA = False, zerointc = False, animal = 'grom'):

        ####### Get cursor ########
        keep_offset = True
        self.curs = Experiment_Cursor(day, keep_offset, animal = animal)
        
        #### Set keep-offset ###### So target location has offset too 
        self.keep_offset = keep_offset


        ####### Get brain ######
        self.brain = NHPBrain_RidgeOffsDyn(day, animal, shuffle, with_intercept = with_intercept,
            state_noise = state_noise, zeroA = zeroA, zerointc = zerointc)

        ###### Get a combined state #####
        # Combined states: 
        self.combostate = ComboStateRidgeOffs(self.curs, self.brain)
        
        self.brain.brain_target = np.zeros((self.brain.nstates)) #self.brain_target; 
        
        if with_intercept:
            self.brain.brain_target[-1] = 1.

        # Now setup the LQR: 
        self.setup_lqr(task, R)

        

######### Get saved ridge regression ########

def get_saved_RidgeOffs(day=0, animal='grom', shuffle = False, with_intercept = True, zeroA = False, zerointc = False):
    
    if zerointc: assert(with_intercept == True)

    #### Try loading from saved place ####
    fname = analysis_config.config['lqr_sim_saved_dyn'] + '%s_%d_shuff%s_wintc%s.pkl' %(animal, day, str(shuffle), str(with_intercept))
    
    if os.path.exists(fname):
        dat = pickle.load(open(fname, 'rb'))
        A = np.mat(dat['A'])
        mu = np.mat(dat['mu'])
        cov = np.mat(dat['cov'])

        if with_intercept:
            offs = np.mat(dat['offs'])
    
    else:
        dyn_model = 'hist_1pos_0psh_0spksm_1_spksp_0'
        if with_intercept:
            dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(6), 'rb'))
        else:
            dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N_no_intc.pkl' %(6), 'rb'))

        if shuffle:
            ##### Load shuffled data ######
            try:
                _, coefs, intc = plot_generated_models.get_shuffled_data(animal, day, dyn_model, get_model = True, 
                with_intercept = True)

                if day == 2:
                    ix = 1; 
                else: 
                    ix = 0; 

                A = np.mat(coefs[ix]); 
                print('True shuffle...')
                if with_intercept:
                    print('with offset!')
                    offs = np.mat(intc[ix][:, np.newaxis]);

            except:
                print('Shuffles not accessible, zeros instead')
                N = dat[day, 'spks'].shape[1]
                A = np.mat(np.zeros((N, N)))
                offs = np.mat(np.zeros((N, 1)))

            ###### offs = mFR #####
            #offs = np.mean(dat[day, 'spks'], axis=0)
            #A = np.mat(np.zeros((offs.shape[0], offs.shape[0])))
            #offs = np.zeros_like(offs)
            #offs = np.mat(offs[:, np.newaxis])
            
            ##### Still get the same info for brain target #####
            mu = np.mat(np.mean(dat[day, 'spks'], axis=0)[:, np.newaxis])
            cov = np.cov(dat[day, 'spks'].T)
            assert(cov.shape[0] == A.shape[0])

        else:
            ##### Load data #####
            #dat = pickle.load(open(analysis_config.config[animal+'_pref'] + 'tuning_models_'+animal+'_model_set%d_task_spec_pls_gen_match_tsk_N.pkl' %(7), 'rb'))
            
            i_m = 2; #### General dynamics; 
            n_folds = 5; 
            i_f = 0; 
            
            model = dat[day, dyn_model, n_folds*i_m + i_f, i_m, 'model']
            A = np.mat(model.coef_)
            
            if with_intercept:
                offs = np.mat(model.intercept_[:, np.newaxis])

            ##### Still get the same info for brain target #####
            mu = np.mat(np.mean(dat[day, 'spks'], axis=0)[:, np.newaxis])
            cov = np.mat(np.cov(dat[day, 'spks'].T))
            assert(cov.shape[0] == A.shape[0])            

        if with_intercept:
            params = dict(A=A, offs=offs, mu=mu, cov=cov)
        else:
            params = dict(A=A, mu=mu, cov=cov)
        
        pickle.dump(params, open(analysis_config.config['lqr_sim_saved_dyn'] + '%s_%d_shuff%s_wintc%s.pkl' %(animal, day, str(shuffle), str(with_intercept)), 'wb'))

    if zeroA: 
        A = np.mat(np.zeros_like(A))
    if with_intercept:
        if zerointc:
            offs = np.mat(np.zeros_like(offs))
        return A, offs, mu, cov
    else:
        return A, mu, cov

