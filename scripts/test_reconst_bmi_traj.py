import dbfunctions as dbfn
from tasks.point_mass_cursor import CursorPlantWithMass
from riglib.plants import CursorPlant
import numpy as np
import multiprocessing as mp
import pickle

class RerunDecoding(object):
    
    def __init__(self, hdf, decoder, task='point_mass', drives_neurons = 0):
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
        self.drives_neurons = self.dec.drives_neurons;
        self.drives_neurons_ix0 = np.nonzero(self.drives_neurons)[0][0]
        self.update_bmi_ix = np.nonzero(np.diff(np.squeeze(self.internal_state[:, self.drives_neurons_ix0, 0])))[0]+1
            
        if task=='point_mass':
            self.plant = CursorPlantWithMass(endpt_bounds=(-14, 14, 0., 0., -14, 14))
            self.move_plant = self.move_mass_plant
            
        elif task == 'bmi_multi':
            self.plant = CursorPlant(endpt_bounds=(-25, 25, 0., 0., -14, 14))
            self.move_plant = self.move_vel_plant
            self.task_msgs = hdf.root.task_msgs

        elif task == 'bmi_resetting':
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

        for t in range(T):
            spike_accum = spike_accum+spike_counts[t,:]
            spike_accum_true = spike_accum_true + spike_counts_true[t, :]
            
            if t in self.task_msgs[:]['time']:
                ix = np.nonzero(self.task_msgs[:]['time']==t)[0]
                self.state = self.task_msgs[ix[0]]['msg']

            if t in self.update_bmi_ix:
                if input_type != 'all':
                    dec_new = self.dec.predict(spike_accum)#/6.)
                else:
                    dec_new = self.dec.predict(spike_accum)

                if self.task == 'bmi_multi':
                    pos = dec_new[[0,1,2]]
                    vel = dec_new[[3,4,5]]
                    pos1, vel1 = self.plant._bound(pos, vel)
                    dec_new[[0,1,2]] = pos1
                    dec_new[[3,4,5]] = vel1
                    self.dec.filt.state.mean = np.array([np.hstack((pos1, vel1, np.array([1.])))]).T

                if self.task == 'bmi_resetting':
                    if self.state == 'premove':
                        self.plant.set_endpoint_pos(np.array([0., 0., 0.]))
                        self.dec['q'] = self.plant.get_intrinsic_coordinates()
                        pos1 = np.array([0., 0., 0.])
                        vel1 = dec_new[[3,4,5]]
                    else:
                        pos = dec_new[[0,1,2]]
                        vel = dec_new[[3,4,5]]
                        pos1, vel1 = self.plant._bound(pos, vel)
                    
                    if save_only_innovation:
                        dec_new[[0, 1, 2]] = np.zeros((3, ))
                        g = np.mat(self.kalman_gain)*np.mat(spike_accum)
                        dec_new[[3, 4, 5]] = np.squeeze(np.array(g[[3, 4, 5], 0]))
                        spike_accum = np.zeros_like(spike_counts[0,:])

                        dec_old = np.zeros((7, 1))
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


#     