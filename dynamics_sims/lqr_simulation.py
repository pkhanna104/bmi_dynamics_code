import numpy as np
import feedback_controllers
import obstacle_fcns
import scipy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pickle
import tables

try:
    import test_smoothness_mets
    import prelim_analysis as pa
except:
    print('Not importing test_smoothness_mets and pa. Its fine, they are not vital')

import analysis_config

### Main Functions ###
### Main Classes: 
    ### Cursor / Experiment_Cursor -- classes that implement a BMI decoder
    ### The Experiment_Cursor uses the real Kalman Gain from the decoder
    ### from a specified day / animal 

    ### Brain / NHPBrain -- implments neural dynamics. 
    ### Brain implements an LDS with a specified number of states / inputs
    ### and computes A such that the eig values all have the same decay 
    ### [value from 0-1], that is the abs(eig_value) and same frequency of rotation

    ### NHPBrain , same as brain but estimates A, C from data

    ### ComboState sets up the combined state [brain, cursor] but isn't that useful 
    ### on its own

    ### Combined_Curs_Brain_LQR_Simulation -- this is the main function that will be 
    ### used for simulation -- this uses a combined Cursor/Brain state, and takes 
    ### LQR params, noise params, task selection (CO or OBS) as an input and has methods
    ### to run task sims.  

    ### Combined_Curs_Brain_LQR_Simulation_Data_Driven -- same as above, but uses the 
    ### NHPBrain/Experiment_Cursor states

### Main Functinos used for simulation: 

### Everything in this section: 
        ########################################################
        ### LDS tests with simulated neural dynamics to plot ###
        ########################################################
### is just pure simulation. I was using these to validate the smoothness metrics,
### to figure out what the heck was up with Jonathan Kao's eigenvalue metrics, 
### and to generate plots showing norm(u) as a fcn of dyanmics. 

### Everything in this section: 
        #################################################################
        ### LDS tests with data-derived neural dynamics/cursor to plot ##
        #################################################################
### uses real neural dynamics (i.e. from data, and using the real Kalman Gain matrix)

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 
'steelblue', 'midnightblue', 'darkmagenta', 'k']

pref = analysis_config.config['grom_pref']

grom_input_type = [[[4377], [4378, 4382]], [[4395], [4394]], [[4411], [4412, 4415]], [[4499], 
[4497, 4498, 4504]], [[4510], [4509, 4514]], [[4523, 4525], [4520, 4522]], [[4536], 
[4532, 4533]], [[4553], [4549]], [[4558], [4560]]]

data_LDS = dict()
#data_LDS['grom', 0] = dat = pickle.load(open(pref+'gromLDSmodels_nstates_%d_combined_models_w_dyn_inno_norms.pkl' %15, 'rb'))

class Cursor(object):
    '''
    cursor evolves according to state: 
    x{t+1} = Ax{t} + Du{t}

    for position terms this looks like: 
        pos{t+1} = pos{t} + dt*vel{t}

    for velocity terms this looks like: 
        vel{t+1} = 0.5*vel{t} + Du{t}
    '''

    def __init__(self, ninputs = 'state_matched', keep_offset = True):

        if keep_offset:
            self.state = np.mat([0., 0., 0., 0., 1.]).reshape(-1,1) # pos x, pos y, vel x, vel y, offset
        else:
            self.state = np.mat([0., 0., 0., 0.]).reshape(-1,1) # pos x, pos y, vel x, vel y
        
        self.keep_offset = keep_offset
        self.nstates = self.state.shape[0]

        if ninputs == 'state_matched':
            self.ninputs = self.nstates;
        else:
            self.ninputs = ninputs 

        # Binsize: 
        self.dt = .1

        ## Input matrix --> think of this as the Kalman Gain: 
        self.B = np.zeros((self.nstates, self.ninputs))

        # Zero out position effects: 
        #self.B[[0, 1], :] = 0; 

        # Edit changed 5/26/20 --> accounted for instantaneous position update; 
        # Only the first 2 input states affect the cursor evolution; 
        self.B[0, 0] = self.dt
        self.B[1, 1] = self.dt

        # Make velocity direct read-out of input: 
        self.B[2, 0] = 1; 
        self.B[3, 1] = 1; 

        # Dynamics matrix: 
        self.A = np.eye(self.nstates)
        self.A[0, 0] = self.A[1, 1] = 1. # Position is persistent
        self.A[0, 2] = self.A[1, 3] = 0.7*self.dt
        self.A[2, 2] = self.A[3, 3] = 0.7 # Velocity decay

        # Add some random, small offset #
        # Added 5/26/20 #
        if self.keep_offset:
            
            #offs = np.random.randn(2, )*.05; 
            offs = np.zeros((2, ))
            self.A[[2, 3], 4] = offs.copy()
            self.A[[0, 1], 4] = self.dt*offs.copy()

        ## Cursor "inputs" are neural state
        self.input_type = 'state'

    def get_next_state(self, input1):
        # Cast input correctly: 
        inp = np.mat(input1).reshape(-1,1)
        ns = np.dot(A, self.state) + np.dot(self.B, inp)
        self.state = ns.copy(); 

class Experiment_Cursor(Cursor):
    '''
    take in a real Kalman filter and get the SSKF matrices
        inputs: kfdecoder is a BMI3D object -- decoder

    open question -- how to generate B? 
    '''

    def __init__(self, day, keep_offset, animal = 'grom'): 

        if animal != 'grom':
            raise Exception('Dont have decoders from %s' %animal)

        ### Run init from above
        super(Experiment_Cursor, self).__init__(keep_offset = keep_offset)

        ### Load up the dictionary of hdf files / decoders.
        ### Use this to overwrite the A and B matrices

        # Get a te_num for day: 
        te_num = grom_input_type[day][0][0]

        pref = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
        co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
        
        dec = co_obs_dict[te_num, 'dec']
        decix = dec.rfind('/')

        ## Load the decoder
        decoder = pickle.load(open(pref+dec[decix:]))
        F, KG = decoder.filt.get_sskf()

        if keep_offset:
            ### Get "input" matrix: 
            ix_keep = [0, 2, 3, 5, 6] # xpos, y pos, xvel, yvel, offset
        else:
            ix_keep = [0, 2, 3, 5]

        self.keep_offset = keep_offset

        ### This is not exactly "A", this is (I-KC)*A
        self.A = F[np.ix_(ix_keep, ix_keep)]

        ### Now get K: This is 5 x 44 (inputs)
        self.B = KG[ix_keep, :]

        ## Cursor "inputs" are neural observations = brain.C*brain.state
        self.input_type = 'obs'

class Brain(object):
    def __init__(self, nstates, ninputs, state_noise_weight, eig_decay, freq, brain_offset = False):
        
        # Assign:
        self.ninputs = ninputs
        self.eig_decay = eig_decay
        self.brain_offset = brain_offset
        if self.brain_offset:
            self.nstates = nstates + 1
        else:
            self.nstates = nstates

        # Get A: 
        self.A = self.create_A(nstates, eig_decay, freq)

        assert(self.A.shape[0] == self.A.shape[1] == self.nstates)

        # Get B: 
        self.B = self.create_B(nstates, ninputs)

        assert(self.B.shape[0] == self.A.shape[1] == self.nstates)

        # Get W: 
        self.W = np.eye(self.nstates)*state_noise_weight

        if self.brain_offset:
            self.W[-1, -1] = 0.

        # Get initial state: 
        self.state = np.mat(np.zeros((nstates, 1))).reshape(-1, 1)
        
        if self.brain_offset:
            self.state = np.vstack((self.state, [1]))

    def create_A(self, nstates, decay, ang_frequency):
        ### ang_frequency --> in degrees
        ### Keep the starting A as eye: 
        As = []
        for i in range(nstates/2):
            omega = ang_frequency * np.pi / 180. + 0.0001*np.random.randn()

            A1 = decay*np.array([[np.cos(omega), -1*np.sin(omega)],
                  [np.sin(omega), np.cos(omega)]])
            As.append(A1)

        A = scipy.linalg.block_diag(*As)
        
        if self.brain_offset:
            A = np.vstack((np.hstack((A, np.zeros((len(A))))), np.zeros((len(A)+1))))
            A[-1, -1] = 1.

        
        # ### What is A, and W? 
        # ### add some noise so we dont have repeated eigenvalues
        # rotate_rads = ang_frequency * np.pi / 180. + (np.random.randn(nstates)*.0001)

        # ### Set the eigenvalues: 
        # diag = np.ones((nstates)).astype(complex)

        # ### Also need to construct the eigenvectors: 

        # for i in range(nstates/2):
        #     diag[(i*2)]   = np.complex(decay*np.cos(rotate_rads[i])   , decay*np.sin(rotate_rads[i]))
        #     diag[(i*2)+1] = np.complex(decay*np.cos(rotate_rads[i]), -1*decay*np.sin(rotate_rads[i]))
        
        # # Do eigenvalue decomposition
        # eigv, eigvect = np.linalg.eig(A)
        # estA =  np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real

        # import pdb; pdb.set_trace()
        # assert(np.all(np.isclose(estA, A)))

        # # now change eigenvalue: 
        # eigv[:] = diag.copy()

        # # New A
        # newA = np.dot(eigvect, np.dot(np.diag(eigv), np.linalg.inv(eigvect))).real

        # return newA

        return A

    def create_B(self, nstates, ninputs):
       #  BOG = np.mat([[-0.13211327,  0.29783435,  1.14589304, -0.1652874 , -0.71903103,
       #   0.92854455, -0.7950932 ,  1.53089965,  1.8831836 , -0.18134923],
       # [-0.87682461, -1.41342625,  1.25023754,  0.41919956, -0.34199317,
       #  -0.71315772,  2.02370016,  0.73728559, -2.18522094,  0.64311617],
       # [ 0.38276929, -0.77739654, -0.10255411,  0.79022097, -0.95347248,
       #  -0.12845421,  0.84671978, -1.26013726, -1.42282342,  0.46912618],
       # [ 0.95939987, -0.86591338,  0.50661693,  1.57258966, -1.3440669 ,
       #   1.13845524,  1.4242924 ,  1.49649354, -0.26424202,  0.78422555],
       # [ 1.07993293,  0.31128315, -1.16638457, -0.40158642, -0.42811412,
       #  -1.17286806,  0.7736843 , -0.83156359,  0.47478318,  0.57181908],
       # [ 0.77590972, -1.4481705 ,  1.51519541, -0.28299535, -0.04909953,
       #  -1.78531243, -0.69665544,  1.56598901,  1.76141345, -1.52231669],
       # [-0.25335444,  0.08580128,  0.69744354,  1.01313763,  0.90183737,
       #   0.78481181,  1.14794529,  0.25603803,  0.60153226,  2.06833593],
       # [ 0.42245994, -1.01709546,  0.36508576,  0.20054975, -0.4199726 ,
       #   0.30415019,  0.47830845, -1.05836246,  0.01032231, -0.09861029],
       # [-1.08732358,  0.01896562, -0.64460712,  0.57610767,  1.42054429,
       #  -1.04806997, -1.43913175, -0.36991963,  1.7289005 , -0.07594465],
       # [ 0.03507117, -1.42765463,  0.98786542, -1.14223858,  1.21651691,
       #  -0.54965896,  1.62663999, -0.18069828,  0.53700523,  1.20915207]])
    
        BOG = np.mat(np.eye(np.max([nstates, ninputs])))

        # Subselect B: 
        B = BOG[np.ix_(np.arange(nstates), np.arange(ninputs))]

        # Make influence of B on neural activity a unit vector 
        B = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

        if self.brain_offset:
            B = np.vstack((B, np.zeros((ninputs, 1)) ))

        return B

    def get_next_state(self, input1):
        inp = np.mat(input1).reshape(-1, 1)
        next = np.dot(self.A, self.state) + np.dot(self.B, inp) + np.random.multivariate_normal(
            np.zeros((self.nstates)), self.W)
        self.state = next.copy()

class NHPBrain(Brain):
    ''' 
    Estimate neural dynamics from data. 
    Use this to set the A and C matrices

    Inputs --> u_t can be any size
    Outputs --> self.state must be state. 

    need to set self.C, self.A, self.B, state noise
    '''
    def __init__(self, ninputs, day = 0, animal = 'grom', state_noise_weight = 0.,
        zeroA = False, modA = None):

        if zeroA: 
            print('Testing, fix this later')
            C = np.random.randn(44, 20)
            #import pdb; pdb.set_trace()
        else:
            A, C = get_saved_LDS(day = day, animal = animal)
            
        if zeroA:
            #self.A = self.create_A(A.shape[0], 0., 0.)
            self.A = self.create_A(20, 0., 0.)
        else:
            self.A = A.copy()

        if type(modA) is np.ndarray: 
            self.A = np.mat(modA.copy())

        if np.logical_and(zeroA, modA is not None):
            raise Exception('Cant ask for a zero A and provide a modified A too! One has got to go')

        self.C = C.copy()

        self.nobs = self.C.shape[0] #
        self.nstates = self.A.shape[0]

        if ninputs == 'state_matched':
            self.ninputs = self.nstates
        elif type(ninputs) is int:
            self.ninputs = ninputs
        else:
            raise Exception('Dont recognize type of inputs arg %s' %str(ninputs))

        self.B = self.create_B(self.nstates, self.ninputs)

        self.W = np.eye(self.nstates)*state_noise_weight

        # Initial state
        self.state = np.mat(np.zeros((self.nstates, 1))).reshape(-1, 1) 

    def get_next_state(self, input1):
        super(NHPBrain, self).get_next_state(input1)
        self.obs = np.dot(self.C, self.state)

class ComboState(object):
    def __init__(self, cursor, brain, add_noise = True, noise_scale = 0.0):
        
        ### Setup the combined neural-cursor state: 
        if cursor.input_type == 'state': 

            ### If cursor inputs are the neural state:
            self.A = np.block([
                [brain.A, np.zeros((brain.nstates, cursor.nstates))],
                [np.dot(cursor.B, brain.A), cursor.A] ])

            self.B = np.block([[ brain.B ],
                              [ np.dot(cursor.B, brain.B)]])

        elif cursor.input_type == 'obs': 

            ### If cursor inputs are the neural observations, need to include brain.C
            self.A = np.block([
                [brain.A, np.zeros((brain.nstates, cursor.nstates))],
                [np.dot(cursor.B, np.dot(brain.C, brain.A)), cursor.A] ])

            self.B = np.block([[ brain.B],
                              [ np.dot(cursor.B, np.dot(brain.C, brain.B))]])

        #self.W = np.eye((brain.nstates))*noise_scale
        self.W = brain.W; 

        self.state = np.vstack(( brain.state, cursor.state ))
        self.nstates = len(self.state)

        self.brain_state_ix = np.arange(brain.nstates)
        self.cursor_state_ix = np.arange(brain.nstates, brain.nstates + cursor.nstates)

        self.cursor = cursor; 
        self.brain = brain

    def get_next_state(self, input):
        inp = np.mat(input)

        ## This is noise in the neural state: 
        noise = np.random.multivariate_normal(np.zeros((len(self.W))), self.W)
        noise = np.mat(noise).reshape(-1, 1)

        ## If the cursor input is directly neural state, then multiply by cursor B: 
        if self.cursor.input_type == 'state':
            noise = np.vstack(( noise, np.dot(self.cursor.B, noise) ))

        elif self.cursor.input_type == 'obs':
            noise = np.vstack(( noise, np.dot(self.cursor.B, np.dot(self.brain.C, noise))))
        
        next = np.dot(self.A, self.state) + np.dot(self.B, inp) + noise

        ### Update the input: 
        self.state = next.copy()
        
    def reset_state(self):
        self.state = np.zeros_like(self.state)

class Combined_Curs_Brain_LQR_Simulation(object):

    def __init__(self, nstates, ninputs=1, R = 10000, dyn_strength = 0.99,
        state_noise = 0., dyn_freq = 10., task = 'co', keep_offset = False,
        brain_target = None):
        
        # 10 states :
        self.curs = Cursor(nstates, keep_offset)

        # 10 states, 10 inputs, no noise, reasonable dynamics
        self.brain = Brain(nstates, ninputs, state_noise, dyn_strength, dyn_freq)

        # Combined states: 
        self.combostate = ComboState(self.curs, self.brain)

        self.setup_lqr(task, R)

        self.keep_offset = keep_offset

        self.brain_target = brain_target

    def setup_lqr(self, task, R): 
        # Set task: 
        self.task = task

        # Q --> we dont care about brain states in cost function: 
        qDiag = np.zeros(( self.combostate.nstates ))

        ### It is ok that we have the offset in the state vector --> since the state is always has offset equal to 1, so 
        ### error should always be 1. 
        qDiag[self.brain.nstates:] = np.mat(np.ones(self.curs.nstates))
        self.Q = np.diag(qDiag)

        # R --> not really sure what to make this: 
        rDiag = np.zeros(( self.brain.ninputs )) + R
        self.R = np.diag(rDiag)

        # LQR controller: 
        self.lqr = feedback_controllers.LQRController(self.combostate.A, 
            self.combostate.B, self.Q, self.R)

        if self.task == 'obs':
            self.obs_goal_calculator = obstacle_fcns.Obs_Goal_Calc()

    def run_all_targets(self, nreps = 8, rad = 10, ax = None, ax2 = None, plot_traj = False,
        plot_states = False, max_trial_time = 1000, collect_brain_state_list = True,
        subset_targs = np.array([0, 2, 4, 6])):

        if collect_brain_state_list:
            self.brain_state_list = []; 

        targs = np.linspace(0., 2*np.pi, 9)[:-1][subset_targs]
        x = rad*np.cos(targs)
        y = rad*np.sin(targs)

        if plot_traj: 
            if ax is None:
                # Plot trajectories: 
                f, ax = plt.subplots()

            if ax2 is None:
                # Plot PSTH of norm of u
                f2, ax2 = plt.subplots()

        state_ax = None
        
        trl_tm = []; 
        total_u = []; 
        mean_u = []; 

        for it, (xi, yi) in enumerate(zip(x, y)):
            us_all = []; us_max = 0; 

            for rep in range(nreps):
                print('Starting rep: %d' %rep)

                # Target location: 
                target_location = np.array([xi, yi])

                ## Plots: 
                if self.task == 'co':
                    states, us = self.simulate_co_trial(target_location, max_trial_time=max_trial_time)
                
                elif self.task == 'obs':
                    states, us = self.simulate_obs_trial(target_location, max_trial_time=max_trial_time)
                    #self.plot_obs_trial(states, us, target_location)

                if plot_states:
                    ### Plot states;
                    state_ax = plot_neural_traj(states, cmap_list[it], self.combostate.brain.A,
                        ax = state_ax)

                if collect_brain_state_list:
                    self.brain_state_list.append(states)

                ### Plot trial: 
                state_pos = states[self.combostate.cursor_state_ix[:2], :]; 
                trl_tm.append(state_pos.shape[1])

                if plot_traj: 
                    ax.plot(state_pos[0, :].T, state_pos[1, :].T, '.-', color = cmap_list[it],
                        linewidth = .5)

                ### Plot PSTH of Us: 
                uss = np.linalg.norm(us, axis=0)
                us_all.append(uss)
                us_max = np.max([us_max, len(uss)])

                total_u.append(np.sum(uss)); 
                mean_u.append(np.mean(uss)) 

                ### Re-set state: 
                self.combostate.reset_state()

            ### Plot mean/sem of us: 
            us_all_mask = np.zeros((nreps, us_max))
            us_all_mask[:] = np.nan

            for iu, us in enumerate(us_all):
                us_all_mask[iu, :len(us)] = us

            ## Mean / sem: 
            mn = np.nanmean(us_all_mask, axis=0)
            st = np.nanstd(us_all_mask, axis=0)
            st = st / np.sqrt(nreps)

            if plot_traj: 
                ax2.plot(np.nanmean(us_all_mask, axis=0), '-', color=cmap_list[it])
                ax2.fill_between(np.arange(us_all_mask.shape[1]), mn-st, mn+st, 
                    color=cmap_list[it], alpha=0.5)
        
        #if plot_traj: 
            #ax2.set_ylim([0., .1])

        if plot_traj: 
            return ax, ax2, np.hstack((trl_tm)), np.hstack((total_u)), np.hstack((mean_u))
        else:
            return np.hstack((trl_tm)), np.hstack((total_u)), np.hstack((mean_u))

    def simulate_co_trial(self, target_location, target_radius = 1.7, hold_time = 0.2,
        cursor_radius = 0.4, max_trial_time = 1000):

        # Final state
        #### Concatentate target location (x, y) with target velocity (0, 0) and offset: 
        if self.keep_offset:
            state_final = np.mat(np.hstack((target_location, [0., 0., 1.]))).reshape(-1, 1)
        else:
            state_final = np.mat(np.hstack((target_location, [0., 0.]))).reshape(-1, 1)

        # Add zeros for brain state: 
        if self.brain_target is None:
            state_final = np.vstack(( np.zeros((self.brain.nstates, 1)), state_final))
        else:
            assert(len(self.brain_target) == self.brain.nstates)
            state_final = np.vstack(( self.brain_target[:, np.newaxis], state_final))

        # Get a K matrix for this -- infinite time horizon: 
        K = self.lqr.dlqr(self.combostate.A, self.combostate.B, self.Q, self.R)

        # Compute state feedback: 
        state_list = []; 
        u_list = []; 

        in_targ_complete = False
        hold_complete = False
        reset_hold = True
        trl_cnt = -1

        while not np.logical_and(in_targ_complete, hold_complete):
            trl_cnt += 1; 

            state_err = self.combostate.state - state_final; 
            
            # Get input: 
            u = -1*np.dot(K, state_err)

            ## Append: 
            state_list.append(self.combostate.state)
            u_list.append(u)

            ## Propogate; 
            self.combostate.get_next_state(u)

            ## Test if state is complete: 
            complete = self.test_in_target(self.combostate.state[self.combostate.cursor_state_ix],
                target_location, target_radius, cursor_radius)

            ## Target completed: 
            if complete:
                in_targ_complete = True

                ## If haven't held yet: 
                if reset_hold:
                    in_targ_cnt = 1; 
                    reset_hold = False

                ## Else if have held, increment hold counter
                else:
                    in_targ_cnt += 1; 

                ### Test if GTE hold time
                if in_targ_cnt >= int(hold_time / .1):
                    hold_complete = True
            else:
                in_targ_complete = False
                reset_hold = True

            if trl_cnt >= 1000: 
                in_targ_complete = True; 
                hold_complete = True;
                print('Aborting CO trial -- timeout time')

        ### If you've broken away from while loop trial is complete! yay! 

        return np.hstack((state_list)), np.hstack((u_list))

    def simulate_obs_trial(self, target_location,target_radius = 1.7, hold_time = 0.2, 
        obs_location_rad = 4., obs_size = 3., cursor_radius = 0.4, max_trial_time = 1000):

        # Final target state
        target_pos = np.mat(np.hstack((target_location, [0., 0.]))).reshape(-1, 1)

        # Obstacle location: 
        # Get angle: 
        ang = np.arctan2(float(target_pos[1, 0]), float(target_pos[0, 0]))
        obstacle_center = np.mat(np.hstack(([obs_location_rad*np.cos(ang), obs_location_rad*np.sin(ang)]))).reshape(-1, 1)

        # Get a K matrix for this -- infinite time horizon: 
        K = self.lqr.dlqr(self.combostate.A, self.combostate.B, self.Q, self.R)

        # Compute state feedback: 
        state_list = []; 
        u_list = []; 

        in_targ_complete = False
        hold_complete = False
        reset_hold = True
        in_obstacle = False
        init = True
        trl_cnt = -1; 

        while not np.logical_and(in_targ_complete, hold_complete):

            ### Coutn the trial time -- timeout sometime
            trl_cnt += 1
            
            ### Depending on position, get the current state: 
            current_goal = self.obs_goal_calculator.call_from_sims(np.array(target_pos)[:2], 
                np.array(obstacle_center)[:2], self.combostate.state[self.combostate.cursor_state_ix][:2])

            if self.keep_offset:
                state_final = np.vstack(( current_goal, [1] ))
            else:
                state_final = np.vstack(( current_goal, ))

            if self.brain_target is None:
                current_goal_state = np.vstack(( np.zeros((self.brain.nstates, 1)), state_final ))
            else: 
                assert(len(self.brain_target) == self.brain.nstates)
                current_goal_state = np.vstack(( self.brain_target[:, np.newaxis], state_final ))


            # if init: 
            #     init = False
            #     last_current_goal = current_goal.copy()
            # else:
            #     if np.sum(last_current_goal - current_goal) != 0.:
            #         last_current_goal = current_goal.copy()

            ### Get current error: 
            state_err = self.combostate.state - current_goal_state; 
            
            # Compute input: 
            u = -1*np.dot(K, state_err)

            ## Append: 
            state_list.append(self.combostate.state.copy())
            u_list.append(u)

            ## Propogate; 
            self.combostate.get_next_state(u)

            ## Test if accidentally entered obstacle: 
            in_obstacle = self.test_enter_obstacle(self.combostate.state[self.combostate.cursor_state_ix], obstacle_center, obs_size)
            
            ## If in obstacle: 
            if in_obstacle:
                # Reset position back to center position: 
                if self.keep_offset:
                    self.combostate.state[self.combostate.cursor_state_ix] = np.mat(np.vstack(( np.zeros((4, 1)), [1])))
                else:
                    self.combostate.state[self.combostate.cursor_state_ix] = np.mat(np.vstack(( np.zeros((4, 1)) )))
                complete = False
            else:
                ## Test if state is complete: 
                complete = self.test_in_target(self.combostate.state[self.combostate.cursor_state_ix],
                    target_location, target_radius, cursor_radius)

            ## Target completed: 
            if complete:
                in_targ_complete = True

                ## If haven't held yet: 
                if reset_hold:
                    in_targ_cnt = 1; 
                    reset_hold = False

                ## Else if have held, increment hold counter
                else:
                    in_targ_cnt += 1; 

                ### Test if GTE hold time
                if in_targ_cnt >= int(hold_time / .1):
                    hold_complete = True
            else:
                in_targ_complete = False
                reset_hold = True

            if trl_cnt >= 1000: 
                in_targ_complete = True; 
                hold_complete = True;
                print('Aborting OBS trial -- timeout time')

        return np.hstack((state_list)), np.hstack((u_list))

    def plot_obs_trial(self, states, Us, target_loc): 
        ''' 
        single plot to figure out what is going on
        '''
        states = np.array(states)
        Us = np.array(Us)

        f, ax = plt.subplots(ncols = 3, nrows = 2); # one per state , one for "U"
        f2, ax2 = plt.subplots() # Cursor traj + obstacle targets plotted, 

        labs = ['xpos', 'ypos', 'xvel', 'yvel']
        for i, st in enumerate(np.arange(-4, 0)):
            ax[ i / 3, i % 3].plot(states[st, :]); 
            ax[ i / 3, i % 3].set_title(labs[i])

        ax[1, 1].plot(np.linalg.norm(Us, axis=0))
        ax[1, 1].set_title('Us')

        ### Plot the target: 
        ax2 = self.obs_goal_calculator.plot_patches(ax2, target_loc, target_loc/2.)
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])

        ### Plot position trajectory
        ax2.plot(states[-4, :], states[-3, : ], '.-')

        import pdb; pdb.set_trace()

    def test_enter_obstacle(self, current_state, obstacle_location, obs_size):
        current_pos = np.array([current_state[0, 0], current_state[1, 0]])
        centered_cursor_pos = np.abs(current_pos - np.squeeze(np.array(obstacle_location[[0, 1], 0])))
        return np.all(centered_cursor_pos < obs_size/2.)

    def test_in_target(self, state_cursor, target_location, target_radius, cursor_radius):
        # Get position from state_cursor: 
        pos = np.squeeze(np.array(state_cursor[[0, 1]]))

        if np.any(np.isnan(pos)):
            import pdb; pdb.set_trace()

        dist_to_targ = np.sqrt(np.sum((target_location - pos)**2))

        if dist_to_targ <= (target_radius - cursor_radius):
            return True
        else:
            return False

    def test_controlabillity(self): 
        ### Compute the controllability Grammian: 
        n_states = self.combostate.A.shape[0]

        P = []
        for n in range(n_states):
            if n == 0:
                P.append(self.combostate.B)
            else:
                P.append(np.dot(self.combostate.A**n, self.combostate.B))
        P = np.hstack((P))

        ### Want the rank of this to be n: 
        rnk = np.linalg.matrix_rank(P)
        print('Controllability test: n states %d, rank of P: %d' %(n_states, rnk))

class Combined_Curs_Brain_LQR_Simulation_Data_Driven(Combined_Curs_Brain_LQR_Simulation):

    def __init__(self, neural_day, ninputs = 'state_matched', R = 10000, state_noise = 0., 
        task = 'co', zeroA = False, keep_offset = True, modA = None):

        self.curs = Experiment_Cursor(neural_day, keep_offset)

        self.brain = NHPBrain(ninputs, day = neural_day, zeroA = zeroA, modA = modA,
            state_noise_weight = state_noise)

        # Combined states: 
        self.combostate = ComboState(self.curs, self.brain)

        # Now setup the LQR: 
        self.setup_lqr(task, R)

        self.keep_offset = keep_offset


### Extract LDS from TEs ###
def get_saved_LDS(day = 0, animal = 'grom', nstates = 20, zeroA = False):
    if animal != 'grom':
        raise Exception('Havent processed jeevs LDS yet --> fit_LDS.fit_LDS_CO_Obs')

    dat = data_LDS[animal, day]

    ## SO far only day 1 is saved: 
    pylds_model = dat[day]

    ## Print the R2 for both tasks: 
    for i in range(2):
        print('Task %d: R2 smooth avg: %.2f' %(i, np.mean(dat[day, i, 'R2_smooth'])))
        print('Task %d: R2 predic avg: %.2f' %(i, np.mean(dat[day, i, 'R2_pred'])))
        print('')
        print('')

    ## Return A and C matrices
    return pylds_model.A, pylds_model.C

########################################################
### LDS tests with simulated neural dynamics to plot ###
########################################################
def sweep_nstates_plot_norm_u(states = np.arange(4, 12, 2)): 
    f, ax = plt.subplots(ncols = len(states), nrows = 2, figsize = (10, 3))

    for i, st in enumerate(states):

        ### Get controller
        ctl =  Combined_Curs_Brain_LQR_Simulation(st)

        ### Plot simulations
        ax0, ax1 = ctl.run_all_targets(nreps=8, rad=8, 
            ax = ax[0, i], ax2 = ax[1, i])

        ### Label stuff: 
        ax0.set_title('N States: %d' %st)
        ax1.set_ylim([0., .1])
        ax1.set_xlim([0., 100.])
    f.tight_layout()

def sims_with_diff_dynamics_strengths(states = 4, inputs = 4,
    dynamics_strength = [0., .25, .5, .75, .9, .99], task = 'co', state_noise = .01): 
    
    ### These aren't really frequencies -- these are angles / timestep ###
    freqs = [0., 5., 10., 15., 20., 30., 40.]
    for freq in freqs:

        ### Plot main figure; 
        f, ax = plt.subplots(ncols = len(dynamics_strength), nrows = 5, 
            figsize = (2*len(dynamics_strength), 8, ))

        for i, dyn in enumerate(dynamics_strength):
            print('Starting dyn: %f' %dyn)

            ### Get controller (offset off by default)
            ctl =  Combined_Curs_Brain_LQR_Simulation(states, inputs, 
                dyn_strength = dyn, task = task, dyn_freq = freq, 
                state_noise = state_noise)

            ### Plot simulations
            ax0, ax1, _, _, _ = ctl.run_all_targets(nreps=1, rad=8, 
                ax = ax[0, i], ax2 = ax[1, i], plot_traj = True)

            ### Collect the brain states: 
            states_brain = [ s[ctl.combostate.brain_state_ix, :].T for s in ctl.brain_state_list]
            
            ### Now fit an LDS on this data: 
            model, r2, dr, ll = fit_LDS_get_states(states_brain)
            detW = '{0:1.2e}'.format(np.linalg.det(model.sigma_states)) 
            ax[2, i].plot(dr)
            ax[3, i].plot(ll)
            ax[4, i].plot(r2)
            ax[3, i].set_title('Det W:'+detW, fontsize=8)
            if i == 0:
                ax[2, i].set_ylabel('Dyn Ratio')
                ax[3, i].set_ylabel('Log Like.')
                ax[4, i].set_ylabel('R2 Pred.')

            ax[2, i].set_ylim([0., 1.1])
            ax[3, i].set_ylim([-2., 7.])
            ax[4, i].set_ylim([0., 1.1])

            for axi in ax[1:, i]:
                axi.set_xlim([0., 1000.])

            ev2, _ = np.linalg.eig(ctl.brain.A)

            # Ms of decay is -1/Re(lambda) * dt, here dt is 100 ms
            angs = np.array([ np.arctan2(np.imag(ev2[i]), np.real(ev2[i])) for i in range(len(ev2))])
            pk_hz = np.abs(angs)/(2*np.pi*.1)

            ### Label stuff: 
            ax0.set_title('Dyn Strength: %.2f, \nHz: %.2f' %(dyn, 
                np.mean(pk_hz)))
            ax1.set_ylim([0., .1])
            ax1.set_xlim([0., 1000.])

            ### Print the eigenvalues of the A matrix
        f.tight_layout()    

### Sweep rotation and dynamics, plot outcomes (trial time, total U, avg. U) ###
def sweep_freq_and_dyn_tasks(states = 4, inputs = 4):
    dynamics_strength = np.linspace(0., .999, 10);
    time_const = -1*0.1*1000 / np.log(dynamics_strength)
    freqs = np.linspace(0., 20., 5);
    
    #Kao Hz vs. Kao decay vs. i) T2T, ii) Total U, iii) avg. U for task CO / OBs
    # PK Hz vs.  PK decay vs. i) T2T, ii) Total U, iii) avg. U
    fkao, axkao = plt.subplots(ncols = 3, nrows = 2, figsize = (9, 6))
    fpk, axpk = plt.subplots(ncols = 3, nrows = 2, figsize = (9, 6))

    X_kao = dict(co=[], obs=[]); 
    X_pk = dict(co=[], obs=[]); 

    for i_t, task in enumerate(['co', 'obs']):

        for i_f, freq in enumerate(freqs):

            for i, dyn in enumerate(dynamics_strength):
                print('Starting dyn: %f' %dyn)

                ### Get controller
                ctl =  Combined_Curs_Brain_LQR_Simulation(states, inputs, 
                    dyn_strength = dyn, task = task, dyn_freq = freq)

                ### Plot simulations
                trl_tm, total_u, mean_u = ctl.run_all_targets(nreps=1, rad=8, 
                    ax = None, ax2 = None)

                ev, _ = np.linalg.eig(ctl.brain.A - np.eye(ctl.brain.nstates))
                ev2, _ = np.linalg.eig(ctl.brain.A)

                # Ms of decay is -1/Re(lambda) * dt, here dt is 100 ms
                kao_ms = -1./np.real(ev)*100.
                angs = np.array([ np.arctan2(np.imag(ev2[i]), np.real(ev2[i])) for i in range(len(ev2))])
                pk_hz = np.abs(angs)/(2*np.pi*.1)
                kao_hz = np.abs(np.imag(ev))/(2*np.pi*.1) 

                tmp = np.zeros_like(trl_tm)

                k = np.vstack(( tmp + kao_ms[0], tmp + kao_hz[0], trl_tm, total_u, mean_u ))
                p = np.vstack(( tmp + dyn,       tmp + pk_hz[0] , trl_tm, total_u, mean_u ))

                X_kao[task].append(k.T)
                X_pk[task].append(p.T)

    X_kao_co = np.vstack((X_kao['co']))
    X_kao_obs = np.vstack((X_kao['obs']))
    X_pk_co = np.vstack((X_pk['co']))
    X_pk_obs = np.vstack((X_pk['obs']))

    X_all = np.hstack(( X_kao_co[:, 2], X_kao_obs[:, 2] ))
    t2t = [np.percentile(X_all, 5), np.percentile(X_all, 95)]

    X_u = np.hstack(( X_kao_co[:, 3], X_kao_obs[:, 3] ))
    totu = [np.percentile(X_u, 5), np.percentile(X_u, 95)]

    X_uu = np.hstack(( X_kao_co[:, 4], X_kao_obs[:, 4] ))
    uu = [np.percentile(X_uu, 5), np.percentile(X_uu, 95)]

    for i, (quant, ax, fig) in enumerate(zip( [[X_kao_co, X_kao_obs], [X_pk_co, X_pk_obs]], [axkao, axpk], [fkao, fpk])):

        for i_t, task in enumerate(quant): 

            for i_m, (met, vlim) in enumerate(zip(range(2, 5), [t2t, totu, uu])):
                image = ax[i_t, i_m].scatter(task[:, 0], task[:, 1], s=None, c=task[:, 2], cmap = 'jet');#
                    #vmin = vlim[0], vmax = vlim[1])
                fig.colorbar(image, ax = ax[i_t, i_m])

    # for ax in axkao.reshape(-1):
    #     ax.set_xlim([0., 1000.])
    fkao.tight_layout()
    fpk.tight_layout()

    return X_kao_co, X_kao_obs, X_pk_co, X_pk_obs

def contour_plot(X_kao_co, X_kao_obs, X_pk_co, X_pk_obs, met_index = 2):
    '''
    Indices: 2 --> t2t, 
             3 --> total u, 
             4 --> mean u
    '''

    f, ax = plt.subplots(ncols = 2, nrows = 2)

    for ix, (method, X) in enumerate(zip(['Kao', 'PK'], [[X_kao_co, X_kao_obs], [X_pk_co, X_pk_obs]])):
        for it, (task_nm, taskX) in enumerate(zip(['co', 'obs'], X)):
            axi = ax[ix, it]
            axi.set_title(method + ', '+task_nm, fontsize=8)

            # -----------------------
            # Interpolation on a grid
            # -----------------------
            # A contour plot of irregularly spaced data coordinates
            # via interpolation on a grid.

            # Create grid values first.
            if method == 'Kao':
                x = np.log10(taskX[:, 0])
            else:
                x = taskX[:, 0]
            vlim = [30, 470]
            y = taskX[:, 1]
            z = taskX[:, met_index]

            xi = np.linspace(np.min(x), np.max(x), 100)
            yi = np.linspace(np.min(y), np.max(y), 100)

            # Perform linear interpolation of the data (x,y)
            # on a grid defined by (xi,yi)
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            # Note that scipy.interpolate provides means to interpolate data on a grid
            # as well. The following would be an alternative to the four lines above:
            #from scipy.interpolate import griddata
            #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
            cntr1 = axi.contourf(xi, yi, zi, 15, cmap="RdBu_r", vmin=vlim[0], vmax=vlim[1])
            m =  plt.cm.ScalarMappable(cmap='RdBu_r')
            m.set_array(zi)
            m.set_clim(vlim[0], vlim[1])
            f.colorbar(m, ax=axi)

#################################################################
### LDS tests with data-derived neural dynamics/cursor to plot ##
#################################################################

def simulate_real_dyn_cursor(day = 0):

    f, ax = plt.subplots(ncols = 4, nrows = 2)

    for i_t, task in enumerate(['co', 'obs']):

        # Control --> no dynamics: 
        for iz, zeroA in enumerate([False, True]): 

            # Test number of inputs: 
            ### Get controller
            ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, task = task, zeroA = zeroA, 
                R = 10000, keep_offset = False)

            ## Test controlabiliyt: 
            ctl.test_controlabillity()

            ### Plot simulations
            _, _, _, _, _ = ctl.run_all_targets(nreps=1, rad=8, plot_traj = True,
                ax = ax[0, (i_t*2) + iz], ax2 = ax[1, (i_t*2) + iz], max_trial_time = 2000)

            ax[0, (i_t*2) + iz].set_title(task + ', zero A: '+str(zeroA), fontsize=8)

    for axi in ax[1, :]:
        axi.set_xlim([0., 300])
        axi.set_ylim([0., .12])

    for axi in ax[0, :]:
        axi.set_xlim([-11, 11])
        axi.set_ylim([-11, 11])

    f.tight_layout()

def lesion_w_noise(day = 0): 

    f2, ax2 = plt.subplots(ncols = 3, nrows = 4)
    xlab = []; xticks = []; 

    cols = ['k', 'blue']

    for i_n, noise in enumerate(np.linspace(0., .025, 10.)):
        xlab.append(['Noise %.2f', ''])
        xticks.append([i_n, i_n + .4])
        
        master_ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, 
            task = 'co', zeroA = False, 
            R = 10000, keep_offset = False, 
            state_noise = noise)

        ### Generate the A's to progressively test:
        brainA = master_ctl.brain.A; 

        dffs = np.zeros((2, 3, 2));

        for i_t, task in enumerate(['co', 'obs']):

            f, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (6, 6))

            ### Generate the A's to progressively tests:
            xlabel = []; 

            for ia, zeroA in enumerate([True, False]):
                ### Get controller
                if zeroA: 
                    ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, task = task, modA = None, 
                        zeroA = True, R = 10000, keep_offset = False, state_noise = noise)
                else:
                    ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, task = task, modA = brainA, 
                        zeroA = False, R = 10000, keep_offset = False, state_noise = noise)
                
                ### Plot simulations
                _, _, trl_tm, u, mn_u = ctl.run_all_targets(nreps=5, rad=8, plot_traj = True,
                    ax = ax[0, ia], ax2 = ax[1, ia], max_trial_time = 2000)
                
                for i, (axi, met) in enumerate(zip(ax2[i_t, :], [trl_tm, u, mn_u])):
                    axi.bar(i_n + 0.4*ia, np.mean(met), width=0.4, color=cols[ia])
                    axi.errorbar(i_n + 0.4*ia, np.mean(met), np.std(met)/np.sqrt(len(met)),
                        color='k', marker='|')

                    dffs[ia, i, i_t] = np.mean(met)
                    
                ax[0, ia].set_title('Noise %.2f, Zero A %s' %(noise, str(zeroA)), fontsize=6)
            f.tight_layout()

        for i in range(3):
            ax2[2, i].bar(i_n, np.mean(dffs[0, i, :] - dffs[1, i, :]), color='green')


    for i_t, task in enumerate(['CO: ', 'Obs: ']):
        for i, (axi, nm) in enumerate(zip(ax2[i_t, :], ['Trl Time', 'Total U', 'Mean U'])):
            axi.set_title(task + nm, fontsize=6)
            if i_t == 1:
                axi.set_xticks(xticks)
                axi.set_xticklabels(xlab)

    f2.tight_layout()

def lesion_real_dyn(day = 0):

    master_ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, 
        task = 'co', zeroA = False, 
        R = 10000, keep_offset = False)

    ### Generate the A's to progressively test:
    brainA = master_ctl.brain.A; 

    ### In order of decreasing dynamics
    A_list, A_ms, A_hz = get_A_list(brainA, 'slow_to_fast')

    ### Setup the norm u plot: 
    f2, ax2 = plt.subplots(nrows = 2, ncols = 3)

    for i_t, task in enumerate(['co', 'obs']):

        f, ax = plt.subplots(ncols = 5, nrows = 2, figsize = (1.5*5, 3))

        ### Generate the A's to progressively tests:
        xlabel = []; 
        for ia, (A, ms, hz) in enumerate(zip(A_list, A_ms, A_hz)):

            ### Get controller
            ctl = Combined_Curs_Brain_LQR_Simulation_Data_Driven(day, task = task, modA = A, 
                zeroA = False, R = 10000, keep_offset = False)

            ## Test controlabillity: 
            # ctl.test_controlabillity()

            ### Plot simulations
            if ia < 5:
                _, _, trl_tm, u, mn_u = ctl.run_all_targets(nreps=1, rad=8, plot_traj = True,
                    ax = ax[0, ia], ax2 = ax[1, ia], max_trial_time = 2000)
            else:
                trl_tm, u, mn_u = ctl.run_all_targets(nreps=1, rad=8, plot_traj = False,
                    max_trial_time = 2000)

            xlabel.append('Decay %.2d, Hz %.2d' %(ms, hz))
            
            for i, (axi, met) in enumerate(zip(ax2[i_t, :], [trl_tm, u, mn_u])):
                axi.bar(ia, np.mean(met), color='blue', alpha=0.3)
                axi.plot([ia]*len(met) + np.random.randn(len(met))*.01, met, 'k.')
                if i == 0:
                    axi.set_ylim([0., 275])
                elif i == 1:
                    axi.set_ylim([0., 10.])
                elif i == 2:
                    axi.set_ylim([0., .06])

            if ia < 5:
                ax[0, ia].set_title('Mod A %d, ms: %.2f, hz; %.2f' %(ia, ms, hz), fontsize=6)
        
        for axi in ax2[i_t, :]:
            axi.set_xticks(np.arange(len(A_hz)))
            axi.set_xticklabels(xlabel, rotation=90, fontsize = 6)
        f.tight_layout()

        for axi in ax[1, :]:
            axi.set_xlim([0., 300])
            axi.set_ylim([0., .12])

        for axi in ax[0, :]:
            axi.set_xlim([-11, 11])
            axi.set_ylim([-11, 11])


    for i_t, task in enumerate(['CO: ', 'Obs: ']):
        for i, (axi, nm) in enumerate(zip(ax2[i_t, :], ['Trl Time', 'Total U', 'Mean U'])):
            axi.set_title(task + nm, fontsize=8)

    f2.tight_layout()

def get_A_list(A, order):  
    A_list = [A]

    ### Get the PK time decay term: 
    pk_ms, pk_hz, _, _ = characterize_A(A)

    if order == 'fast_to_slow':
        ### Order from short to long
        ix_sort = np.argsort(pk_ms)

    elif order == 'slow_to_fast':
        ix_sort = np.argsort(pk_ms)[::-1]
        
    ### Make sure complex eigs are paired: 
    ix_sort2 = []; ms = []; hz = []; 
    for i in ix_sort: 
        proceed = False
        if len(ix_sort2) > 0:
            if i not in np.hstack((ix_sort2)):
                proceed = True
        else:
            proceed = True

        if proceed:
            tm = pk_ms[i]; 
            ix = np.nonzero(pk_ms==tm)[0]
            ix_sort2.append(ix)
            ms.append(tm)
            hz.append(pk_hz[i])

    ### Eigenvalues and vectors
    ev, T = np.linalg.eig(A)
    L = np.diag(ev)
    TI = np.linalg.inv(T)

    ### Make sure eigenvalue multiplication works out properly: 
    assert(np.allclose(A, np.real(np.dot(T, np.dot(L, TI)))))
    
    ix_agg = []
    for ixs in ix_sort2:
        ix_agg.append(ixs)

        ev_sub = ev.copy()
        ev_sub[np.hstack((ix_agg))] = 0.
        L_sub = np.diag(ev_sub)
        sub_A = np.dot(T, np.dot(L_sub, TI))

        A_list.append(np.real(sub_A))

    return A_list, ms, hz
        
####################################################
#### Match sims to normal neural data (time 2 target)
####################################################

def get_trial_time_dist(te_nums):
    # elif on preeyas MBP
    co_obs_dict = pickle.load(open(pref+'co_obs_file_dict.pkl'))
    

    trial_time_dict = dict()

    for te_num in te_nums:
        hdf = co_obs_dict[te_num, 'hdf']
        hdfix = hdf.rfind('/')
        hdf = tables.openFile(pref+hdf[hdfix:])

        dec = co_obs_dict[te_num, 'dec']
        decix = dec.rfind('/')
        decoder = pickle.load(open(pref+dec[decix:]))
        F, KG = decoder.filt.get_sskf()

        # Get trials: 
        drives_neurons_ix0 = 3
        key = 'spike_counts'
        
        rew_ix = np.array([t[1] for it, t in enumerate(hdf.root.task_msgs[:]) if t[0]=='reward'])

        # decoder_all is now KG*spks
        # decoder_all units (aka 'neural push') is in cm / sec
        bin_spk, targ_i_all, targ_ix, trial_ix_all, decoder_all = pa.extract_trials_all(hdf, rew_ix, 
            drives_neurons_ix0=drives_neurons_ix0, hdf_key=key, keep_trials_sep=True,
            reach_tm_is_hdf_cursor_pos=False, reach_tm_is_kg_vel=True, 
            include_pre_go = 0., **dict(kalman_gain=KG))

        trial_time_dict[te_num] = [b.shape[0]*.1 for b in bin_spk]

    f, ax = plt.subplots()
    cols = ['r', 'g', 'b']
    for it, te_num in enumerate(te_nums):
        hist, edges = np.histogram(trial_time_dict[te_num], bins=15)
        edge_center = edges[:-1] + (edges[1] - edges[0])*.5
        ax.plot(edge_center, hist, cols[it]+'-', label = str(te_num))
        ax.vlines(np.mean(trial_time_dict[te_num]), 0, 35, cols[it])

    plt.legend()

    return trial_time_dict

#############################
### Plotting / utils fcns ###
#############################

def plot_neural_traj(states, color, A, ax = None):
    ''' 
    assumes states is in format nstates x T
    '''
    ### Ignore bottom 4 states --> cursor
    nstates = int(states.shape[0]) - 1
    state_pairs = np.vstack(([[i, i+1] for i in np.arange(0, nstates, 2)]))

    if ax is None:
        f, ax = plt.subplots(ncols = int(nstates/2))

        if nstates == 2:
            ax = [ax]

        ### Label the states: 
        for ip, pair in enumerate(state_pairs):
            axi = ax[ip]

            ### Add labels: 
            axi.set_xlabel('State %d' %pair[0])
            axi.set_ylabel('State %d' %pair[1])

            ### Plot the flow field: 
            axi = plot_flow(axi, pair[0], pair[1], A)

    for ip, pair in enumerate(state_pairs):
        x0, x1 = pair
        axi = ax[ip]

        ## Plot state: 
        axi.plot(states[x0, :].T, states[x1, :].T, '-', color=color, linewidth = .5)

        axi.set_ylim([-.5, .5])
        axi.set_xlim([-.5, .5])
        
    return ax

def plot_flow(ax, s0, s1, A):

    nb_points   = 20
    x = np.linspace(-.5, .5, nb_points)
    y = np.linspace(-.5, .5, nb_points)
    X1 , Y1  = np.meshgrid(x, y)                       # create a grid
    
    ### Get changes on grid: 
    DX, DY = compute_dX(X1, Y1, A, s0, s1)  
    M = (np.hypot(DX, DY))         
    #M[ M == 0] = 1. 

    # Normalize the arrays: 
    #DX /= M
    #DY /= M

    # Draw direction fields, using matplotlib 's quiver function
    # I choose to plot normalized arrows and to use colors to give information on
    # the growth speed
    Q = ax.quiver(X1, Y1, DX, DY, M, pivot='mid', cmap=plt.cm.hot,
        clim = [0., 1.])

    return ax

def compute_dX(X, Y, A, dim1, dim2):
    newX = np.zeros_like(X)
    newY = np.zeros_like(Y)

    nrows, ncols = X.shape

    for nr in range(nrows):
        for nc in range(ncols):
            st = np.zeros((len(A), 1))
            st[dim1] = X[nr, nc]; 
            st[dim2] = Y[nr, nc];

            st_nx = np.dot(A, st)
            newX[nr, nc] = st_nx[dim1]
            newY[nr, nc] = st_nx[dim2]

    ### Now to get the change, do new - old: 
    DX = newX - X; 
    DY = newY - Y; 

    return DX, DY

def plot_all_As():
    f1, ax1 = plt.subplots(ncols = 3, nrows = 3)
    f2, ax2 = plt.subplots(ncols = 3, nrows = 3)

    for _, (nm, ax) in enumerate(zip(['pk', 'kao'], [ax1, ax2])):        
        for day in range(9):
            axi = ax[day / 3, day % 3]

            A, C = get_saved_LDS(day = day)
            plot_A(A, ax = axi, kao_or_pk = nm)

            axi.set_xlim([0., 1500.])
            axi.set_ylim([-5, 5])
            axi.set_title('Day %d' %day)

    f1.tight_layout()
    f2.tight_layout()

def plot_A(A, ax = None, kao_or_pk = 'kao'): 
    ''' 
    for varying levels of decay, diff numbers of states plot the A matrix
    ''' 

    if ax is None:
        f, ax = plt.subplots()

    pk_ms, pk_hz, kao_ms, kao_hz = characterize_A(A)

    if kao_or_pk == 'kao':
        ax.plot(kao_ms, kao_hz, '.')

    elif kao_or_pk == 'pk':
        ax.plot(pk_ms, pk_hz, '.')

def characterize_A(A, dt=0.1):

    ### get eigenvalue of cts system and discrete system
    ev_M, _ = np.linalg.eig((A - np.eye(A.shape[0])))
    ev_A, _ = np.linalg.eig(A)

    ### get time decay by -1./ln(abs(eigvalue_A)) 
    pk_ms = -1./np.log(np.abs(ev_A))*dt*1000. # Time decay constant in ms

    # time units of decay is -1/Re(lambda) --> *.1 to seconds --> * 1000 to ms
    kao_ms = -1./np.real(ev_M)*dt*1000. # Time decay constant of CTS approx matrix in ms

    ## Get angs of A matrix eigenvalues
    angs = np.array([ np.arctan2(np.imag(ev_A[i]), np.real(ev_A[i])) for i in range(len(ev_A))])

    ## Get the frequency: rad / dt --> cycle / dt --> cycle / sec
    pk_hz = angs/(2*np.pi*dt)

    ## get freq from M matrix: 
    kao_hz = np.imag(ev_M)/(2*np.pi*dt) 

    return pk_ms, pk_hz, kao_ms, kao_hz

def eigen_test(dt = .1): 
    f, ax = plt.subplots(ncols = 2, nrows = 2)

    for i_f, freq in enumerate(np.arange(0., 3.0, .5)):
        F = []; D = []; Ns = []; Hz = []; PK_Hz = []; PK_ms = []; 

        for time_const in np.linspace(100., 2000., 10): 
            
            ### Compute R from the time constant
            r = np.exp(-1*dt*1000/time_const)

            rad_per_step = freq*(2*np.pi)*dt; # Cycles/sec * (rad / cycle) * (sec / ts) 
            A = r*np.array([[np.cos(rad_per_step), -1*np.sin(rad_per_step)],
                                [np.sin(rad_per_step), np.cos(rad_per_step)]])
            
            x = np.array([[1.], [0.]])
            xs = [x]
            for j in range(100): 
                x = np.dot(A, x)
                xs.append(x)
            xs = np.hstack((xs))

            ### Get A matrix: 
            pk_ms, pk_hz, kao_ms, kao_hz = characterize_A(A, dt = dt)

            F.append(freq); D.append(time_const); Ns.append(kao_ms[0]); Hz.append(kao_hz[0])
            PK_Hz.append(pk_hz[0]); PK_ms.append(pk_ms[0])
    
        F = np.hstack((F)); D = np.hstack((D)); Ns = np.hstack((Ns)); Hz = np.hstack((Hz))
        PK_Hz = np.hstack((PK_Hz)); PK_ms = np.hstack((PK_ms))


        ax[0, 0].plot(D, Ns, '-', color=cmap_list[i_f], label=str(freq)+' Hz')
        ax[0, 1].plot(D, Hz, '-', color = cmap_list[i_f], label=str(freq)+ ' Hz')
        ax[1, 0].plot(D, PK_ms, '-', color=cmap_list[i_f], label=str(freq)+' Hz')
        ax[1, 1].plot(D, PK_Hz, '-', color = cmap_list[i_f], label=str(freq)+ ' Hz')

    ax[0, 0].set_title('dt of Simulation: %.3f' %dt, fontsize=8)

    ax[1, 0].set_xlabel('Time constant of eigenvalues of A', fontsize = 8)
    ax[1, 1].set_xlabel('Time constant of eigenvalues of A', fontsize = 8)

    ax[0, 0].set_ylabel('Kao ms', fontsize = 8)
    ax[0, 1].set_ylabel('Kao Hz', fontsize = 8)

    ax[1, 0].set_ylabel('-1 / ln(abs(eig(A)))', fontsize = 8)
    ax[1, 1].set_ylabel('Freq -- angle(eig(A)) Hz', fontsize = 8)

    for axi in ax.reshape(-1):
        axi.legend(fontsize=6)

    ax[0, 0].set_ylim([0., 2000.])
    ax[1, 0].set_ylim([0., 2000.])

    f.tight_layout()


#############################
### LDS fcns ################
#############################
def fit_LDS_get_states(data):
    model = test_smoothness_mets.fit_LDS_model(data[0].shape[1], data[0].shape[1], 0, data, init_w_FA = False, nEMiters = 30)

    ### Compute likelihood
    r2, dr, ll = test_smoothness_mets.get_metric_psth(data, model, pre_go_bins=1)

    return model, r2, dr, ll

