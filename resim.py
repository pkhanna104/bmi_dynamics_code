import dbfunctions as dbfn
from tasks.point_mass_cursor import CursorPlantWithMass
from riglib.plants import CursorPlant
import numpy as np
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
import tables
import scipy
import analysis_config
import scipy.io as sio

cmap_list = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 'teal', 
    'steelblue', 'midnightblue', 'darkmagenta']


def test():
    tmbu = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/grom2016/'
    hdf = tables.openFile(tmbu+'grom20160307_04_te4411.hdf')
    dec = pickle.load(open(tmbu+'grom20160307_02_RMLC03071555.pkl'))
    R = RerunDecoding(hdf, dec, task='bmi_resetting')

    sc = hdf.root.task[:]['spike_counts']
    R.run_decoder(sc, False)

    plot_traj(R)

class RerunDecoding(object):
    
    def __init__(self, hdf, decoder, task='bmi_multi', drives_neurons = 0, center = np.array([0., 0., 0.])):
        self.center = center
        try:
            self.cursor_pos = hdf.root.task[:]['cursor_pos']
        except:
            self.cursor_pos = hdf.root.task[:]['cursor']
            
        try:
            self.cursor_vel = hdf.root.task[:]['cursor_vel']
        except:
            self.cursor_vel = hdf.root.task[:]['internal_decoder_state'][:,[3,4,5]]

        self.target = hdf.root.task[:]['target']

        self.target_rad = hdf.root.task.attrs.target_radius
        self.cursor_rad = hdf.root.task.attrs.cursor_radius
        try:
            spike_counts = hdf.root.task[:]['spike_counts']
        except:
            spike_counts = hdf.root.task[:]['all']
            
        self.spike_counts = np.array(spike_counts, dtype=np.float64)
        
        self.internal_state = hdf.root.task[:]['internal_decoder_state']

        self.dec = decoder
        F, K = self.dec.filt.get_sskf()
        self.kalman_gain = np.mat(K)
        self.F = np.mat(F)

        self.drives_neurons = self.dec.drives_neurons;
        self.drives_neurons_ix0 = np.nonzero(self.drives_neurons)[0][0]
        self.update_bmi_ix = np.nonzero( np.round( np.diff(np.squeeze(self.internal_state[:, self.drives_neurons_ix0, 0])), 6))[0]+1
        T = self.internal_state.shape[0]

        #### Seems like "internal state" is not reliable for homer data....
        self.update_bmi_ix = np.arange(self.update_bmi_ix[0], T, 6)

        if task=='point_mass':
            self.plant = CursorPlantWithMass(endpt_bounds=(-14, 14, 0., 0., -14, 14))
            self.move_plant = self.move_mass_plant
            
        elif task == 'bmi_multi':
            self.plant = big_cursor_25x14 = CursorPlant(endpt_bounds=(-6, 14, 0., 0., -10, 8), cursor_radius=1.0)
            #self.plant = CursorPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
            self.move_plant = self.move_vel_plant
            self.task_msgs = hdf.root.task_msgs

        elif task == 'bmi_resetting':
            if np.sum(np.abs(center)) > 0:
                self.plant = CursorPlant(endpt_bounds=(-6, 14, 0., 0., -10, 8), cursor_radius=1.0)
            else:
                self.plant = CursorPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
            self.move_plant = self.move_vel_plant
            self.task_msgs = hdf.root.task_msgs

        self.task = task
        self.assist = hdf.root.task[:]['assist_level']
        self.hdf = hdf

    def run_decoder(self, spike_counts, save_only_innovation, input_type = 'all', cutoff=None, 
        spike_counts_true=None):
        '''
        Summary: method to use the 'predict' function in the decoder object
        Input param: spike_counts: unbinned spike counts in iter x units x 1
        Input param: cutoff:  cutoff in iterations
        '''
        if spike_counts_true is None:
            spike_counts_true = self.spike_counts
        else:
            print 'using spike counts from args'

        T = spike_counts.shape[0]
        if not (cutoff is None):
            T = np.min([T, cutoff])
        
        decoded_state = []
        decoded_state_OG = []
        dec_old = np.zeros((7, ))

        spike_accum = np.zeros_like(spike_counts[0,:])
        spike_accum_true = np.zeros_like(spike_counts[0,:])
        
        dec_last = np.zeros_like(self.dec.predict(spike_counts[0,:]))
        dec_last_og = np.zeros((7, 1))
        
        tot_spike_accum = np.zeros_like(spike_counts[0,:])-1
        if self.task == 'point_mass':
            self.dec.filt.state.mean = np.zeros_like(self.dec.filt.state.mean)
        
        elif (self.task == 'bmi_multi' or self.task == 'bmi_resetting'):
            self.dec.filt._init_state()
            self.state = self.task_msgs[0]['msg']

        cnt = 0
        for t in range(T):
            spike_accum = spike_accum+spike_counts[t,:]
            spike_accum_true = spike_accum_true + spike_counts_true[t, :]
            
            if t in self.task_msgs[:]['time']:
                ix = np.nonzero(self.task_msgs[:]['time']==t)[0]
                self.state = self.task_msgs[ix[0]]['msg']

            if t in self.update_bmi_ix:
                cnt += 1
                if np.sum(spike_accum) > 1000:
                    import pdb; pdb.set_trace()

                pre_state = self.dec.filt.state.mean; 

                if input_type != 'all':
                    dec_new = self.dec.predict(spike_accum)#/6.)
                else:
                    dec_new = self.dec.predict(spike_accum)

                #### Try to explicitly map out dec_new updates ####
                z = (np.asarray(spike_accum.ravel()) - self.dec.mFR)*(1./self.dec.sdFR)
                typical_pred = np.dot(self.F, pre_state) + np.dot(self.kalman_gain, z[:, np.newaxis])
                try:
                    assert(np.allclose(np.asarray(typical_pred).ravel(-1), dec_new))
                except:
                    print('iter %d, error %.2f' %(cnt, np.linalg.norm(np.asarray(typical_pred).ravel(-1) - dec_new)))

                if self.task == 'bmi_multi':
                    pos = dec_new[[0,1,2]]
                    vel = dec_new[[3,4,5]]
                    pos1, vel1 = self.plant._bound(pos, vel)
                    dec_new[[0,1,2]] = pos1
                    dec_new[[3,4,5]] = vel1
                    self.dec.filt.state.mean = np.array([np.hstack((pos1, vel1, np.array([1.])))]).T
                    spike_accum = np.zeros_like(spike_counts[0,:])
                    spike_accum_true = np.zeros_like(spike_counts[0,:])
                
                if self.task == 'bmi_resetting':
                    if self.state == 'premove':
                        self.plant.set_endpoint_pos(self.center)
                        self.dec['q'] = self.plant.get_intrinsic_coordinates()
                        pos1 = self.center
                        vel1 = dec_new[[3,4,5]]
                    else:
                        pos = dec_new[[0,1,2]]
                        vel = dec_new[[3,4,5]]
                        pos1, vel1 = self.plant._bound(pos, vel)
                    
                    if save_only_innovation:
                        dec_new[[0, 1, 2]] = self.center
                        g = np.mat(self.kalman_gain)*np.mat(spike_accum)
                        dec_new[[3, 4, 5]] = np.squeeze(np.array(g[[3, 4, 5], 0]))
                        spike_accum = np.zeros_like(spike_counts[0,:])

                        dec_old = np.zeros((7, 1))
                        dec_old[[0, 1, 2]] = self.center
                        g2 = np.mat(self.kalman_gain)*np.mat(spike_accum_true)
                        dec_old[[3, 4, 5]] = np.squeeze(g2[[3, 4, 5], 0]).T
                        spike_accum_true = np.zeros_like(spike_counts[0,:])
                    
                    else:
                        dec_new[[0,1,2]] = pos1
                        dec_new[[3,4,5]] = vel1
                        spike_accum = np.zeros_like(spike_counts[0,:])
                        spike_accum_true = np.zeros_like(spike_counts[0,:])
                    
                    self.dec.filt.state.mean = np.array([np.hstack((pos1, vel1, np.array([1.])))]).T                    

                tot_spike_accum = np.hstack((tot_spike_accum, spike_accum))
                decoded_state.append(dec_new)

                if save_only_innovation:
                    decoded_state_OG.append(dec_old)
                    dec_last_og = dec_old
               
                dec_last = dec_new

            else:

                if self.task == 'bmi_resetting' and self.state == 'premove' and np.sum(np.abs(self.center))>0:
                    self.plant.set_endpoint_pos(self.center)
                    self.dec['q'] = self.plant.get_intrinsic_coordinates()
                    tmp_dec_last = dec_last.copy()
                    tmp_dec_last[[0, 1, 2]] = self.center
                    decoded_state.append(tmp_dec_last) 
                    # dont remember whtat decoded_state_OG is for homer sims               
                else:
                    decoded_state.append(dec_last)
                    decoded_state_OG.append(dec_last_og)

        spk_cnt = np.array(tot_spike_accum)

        if not hasattr(self, 'dec_spk_cnt_bin'):
            self.dec_spk_cnt_bin = dict()
            self.dec_state_mn = dict()
        print 'input_type: ', input_type
        self.dec_spk_cnt_bin[input_type] = spk_cnt[:,1:]
        self.dec_state_mn[input_type] = np.vstack((decoded_state))
        self.dec_state_mn['true'] = np.hstack((decoded_state_OG)).T
        
    def move_vel_plant(self, reset_ix, dt=1/60.):
        pass

    def move_mass_plant(self, reset_ix = [], dt = 1/60., input_type='all'):
        go_res = 0
        p0 = self.cursor_pos[0,:].copy()
        v0 = self.cursor_vel[0,:].copy()
        
        pos_arr = []
        vel_arr = []

        for i in range(1, self.dec_state_mn[input_type].shape[0]):

            force = self.dec_state_mn[input_type][i-1,[9, 10, 11]]
            vel = v0 + dt*force
            pos = p0 + dt*vel + 0.5*dt**2*force
            pos, vel = self.plant._bound(pos,vel)


            #Check if next index is the start of a trial
            if i+1 in reset_ix:
                p0 = self.cursor_pos[i,:]
                v0 = self.cursor_vel[i,:]
                go_res += 1
                print go_res

            else:
                p0 = pos.copy()
                v0 = vel.copy()

            pos_arr.append(pos)
            vel_arr.append(vel)
        self.decoded_pos[input_type] = np.array(pos_arr)
        self.decoded_vel[input_type] = np.array(vel_arr)

    def add_input(self, spike_counts, input_type, save_only_innovation=False):
        self.run_decoder(spike_counts, input_type=input_type, save_only_innovation=save_only_innovation)
        self.main_move_plant(input_type=input_type)


    def main_move_plant(self, input_type):
        #For Vel BMI: 
        if not hasattr(self, 'decoded_pos'):
            self.decoded_pos = dict()
            self.decoded_vel = dict()

        if self.task == 'bmi_multi':
            go_ix = np.array([self.hdf.root.task_msgs[it-3][1] for it, t in enumerate(self.hdf.root.task_msgs[:]) if t[0] == 'reward'])

            self.decoded_pos[input_type] = self.dec_state_mn[input_type][:,[0,1,2]]
            self.decoded_vel[input_type] = self.dec_state_mn[input_type][:,[3,4,5]]

            for g in go_ix:
                if g < self.decoded_pos[input_type].shape[0]:
                    p0 = self.cursor_pos[g,:]
                    dp = p0 - self.decoded_pos[input_type][g,:]
                    self.decoded_pos[input_type][g:,:] = self.decoded_pos[input_type][g:,:] + np.tile(np.array([dp]), [ self.decoded_pos[input_type][g:,:].shape[0],1])
        
        elif self.task == 'bmi_resetting':
            self.decoded_pos[input_type] = self.dec_state_mn[input_type][:,[0,1,2]]
            self.decoded_vel[input_type] = self.dec_state_mn[input_type][:, [3,4,5]]
        
        else:
            go_ix = np.array([self.hdf.root.task_msgs[it-3][1] for it, t in enumerate(self.hdf.root.task_msgs[:]) if t[0] == 'reward'])
            self.move_plant(reset_ix = list(go_ix))

def plot_traj(R, plot_pos=1, plot_vel=1, plot_force=0, it_cutoff=20000, 
    min_it_cutoff=0, input_type='all', monk=''):
    
    rew_ix = np.array([i[1] for i in R.hdf.root.task_msgs[:] if i[0]=='reward'])
    go_ix = np.array([R.hdf.root.task_msgs[it-3][1] for it, t in enumerate(R.hdf.root.task_msgs[:]) if t[0] == 'reward'])
    
    #Make sure no assist is used 
    try:
        zero_assist_start = np.nonzero(R.hdf.root.task[:]['assist_level']==0)[0][0]
    except:
        print 'ignoring assist'
        zero_assist_start = 0

    keep_ix = scipy.logical_and(go_ix>np.max([min_it_cutoff, zero_assist_start+(60*60)]), go_ix<it_cutoff)
    go_ix = go_ix[keep_ix]

    rew_ix = rew_ix[keep_ix]

    targ_pos = R.hdf.root.task[go_ix.astype(int)+5]['target']

    if monk == 'homer':
        targ_ix = get_target_ix_homer(targ_pos[:, [0, 2]]).astype(int)
    else:
        targ_ix = get_target_ix(targ_pos[:,[0, 2]]).astype(int)

    if plot_pos == 1:
        f, ax = plt.subplots(nrows=4, ncols=2)
        f2, ax2 = plt.subplots(nrows=4, ncols=2)
        f0, ax0 = plt.subplots()
    
    if plot_vel == 1:
        f3, ax3 = plt.subplots(nrows=4, ncols=2)
        f4, ax4 = plt.subplots(nrows=4, ncols=2)        

    if plot_force ==1:
        f5, ax5 = plt.subplots(nrows=4, ncols=2)
        f6, ax6 = plt.subplots(nrows=4, ncols=2)
    #R = pickle.load(open(pkl_name))

    for i, (g,r) in enumerate(zip(go_ix, rew_ix)):
        #Choose axis: 
        targ = targ_ix[i]
        
        if plot_pos == 1:
            axi = ax[targ%4, targ/4]
            axi2 = ax2[targ%4, targ/4]

            axi.plot(R.cursor_pos[g:r, 0], 'k-')
            axi2.plot(R.cursor_pos[g:r, 2], 'k-')

            axi.plot(R.dec_state_mn[input_type][g:r, 0], 'b-')
            axi2.plot(R.dec_state_mn[input_type][g:r, 2], 'b-')
            
            ax0.plot(R.cursor_pos[g:r,0], R.cursor_pos[g:r, 2], '-', color='k')
            ax0.plot(R.dec_state_mn[input_type][g:r,0], R.dec_state_mn[input_type][g:r, 2], '--', color=cmap_list[int(targ)])


            if i==0:
                axi.set_title('Position X')
                axi2.set_title('Position Z')

        if plot_vel == 1:
            axi3 = ax3[targ%4, targ/4]
            axi4 = ax4[targ%4, targ/4]
            axi3.plot(R.cursor_vel[g:r, 0], 'k-')
            axi4.plot(R.cursor_vel[g:r, 2], 'k-')

            axi3.plot(R.dec_state_mn[input_type][g:r, 3], 'b-')
            axi4.plot(R.dec_state_mn[input_type][g:r, 5], 'b-')
            if i==0:
                axi3.set_title('Vel X')
                axi4.set_title('Vel Z')

        if plot_force ==1:
            axi5 = ax5[targ%4, targ/4]
            axi6 = ax6[targ%4, targ/4]
            axi5.plot(R.dec_state_mn[input_type][g-1:r-1, 9], 'b-')
            axi6.plot(R.dec_state_mn[input_type][g-1:r-1, 11], 'b-')

            axi5.plot(hdf.root.task[g:r]['internal_decoder_state'][:,9], 'k-')
            axi6.plot(hdf.root.task[g:r]['internal_decoder_state'][:,11], 'k-')
            if i==0:
                axi5.set_title('Acc X')
                axi6.set_title('Acc Z')

    plt.tight_layout()

def get_target_ix(targ_pos):

    dats = sio.loadmat(analysis_config.config['grom_pref'] + 'unique_targ.mat')
    unique_targ = dats['unique_targ']

    targ_ix = np.zeros((targ_pos.shape[0]), )
    for ig, (x,y) in enumerate(targ_pos):
        tmp_ix = np.nonzero(np.sum(targ_pos[ig,:]==unique_targ, axis=1)==2)[0]
        if len(tmp_ix) > 0:
            targ_ix[ig] = tmp_ix
        else:
            targ_ix[ig] = -1

    return targ_ix

def get_target_ix_homer(targ_pos):

    unique_targ = np.array([[0.40380592, -5.59619408],
                            [5., -7.5],
                            [9.59619408, -5.59619408],
                            [11.5, -1.],
                            [9.59619408, 3.59619408],
                            [5., 5.5],
                            [0.40380592, 3.59619408],
                            [-1.5, -1]])

    targ_ix = np.zeros((targ_pos.shape[0]), )
    for ig, (x,y) in enumerate(targ_pos):

        dist = np.linalg.norm(unique_targ - targ_pos[ig, :][np.newaxis, :], axis=1)
        assert(len(dist) == unique_targ.shape[0])

        ix = np.argmin(dist)
        if dist[ix] > .5:
            targ_ix[ig] = -1
        else:
            targ_ix[ig] = ix
    return targ_ix
