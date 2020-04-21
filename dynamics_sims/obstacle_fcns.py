import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

class Obs_Goal_Calc(object):
    '''
    Method copied / cleaned up from BMI3D goal_calculators.Obs_Goal_Calc. 
    Functionally identical to the goal estimator used in CLDA Obstacle tasks
    '''
    def __init__(self, ssm=None, **kwargs):
        self.ssm = ssm
        self.mid_speed = kwargs.pop('mid_targ_speed', 10)
        self.mid_targ_rad = kwargs.pop('mid_targ_rad', 6)

    def __call__(self, target_pos, obstacle_center, current_pos):
  
        center = np.zeros((3, ))
        target_pos = target_pos.round(1)

        ### Are we closer to the center or the target?     
        pre_obs = self.fcn_det(current_pos, center, target_pos)

        # If we are pre-obstacle, aim for the intermediate target
        if pre_obs:
            ## Theta of obstacle angle: 
            obs_ang = np.angle(obstacle_center[0]-center[0] + 1j*(obstacle_center[2]-center[2]))

            # Radius of obstacle 
            obs_r = np.abs((obstacle_center[0]-center[0]) + 1j*(obstacle_center[2]-center[2]))
            
            # which side of the obstacle are we on right now? Return "true" for CCW, "false" for CW
            if self.ccw_fcn(current_pos, obstacle_center): 
                targ_vect_ang = np.pi/2
            else:
                targ_vect_ang = -1*np.pi/2

            ## Intermediate target position: 
            target_state_pos = obstacle_center + self.mid_targ_rad*(np.array([np.cos(targ_vect_ang+obs_ang), 0, np.sin(targ_vect_ang+obs_ang)]))

            # Intermediate target velocity: 
            target_vel = self.mid_speed*np.array([np.cos(obs_ang), 0, np.sin(obs_ang)])
            target_state = np.hstack((target_state_pos, target_vel, 1)).reshape(-1, 1)
            
        # If we are not, then aim for the final target: 
        else:
            target_vel = np.zeros_like(target_pos)
            offset_val = 1
            target_state = np.hstack([target_pos, target_vel, 1]).reshape(-1, 1)
        return target_state

    def call_from_sims(self, target_pos, obstacle_center, current_pos):
        ### Add teh zeros in: 
        target_pos_inf = np.array([target_pos[0, 0], 0., target_pos[1, 0], ])
        obstacle_center_inf = np.array([obstacle_center[0, 0], 0., obstacle_center[1, 0]])
        current_pos_inf = np.array([current_pos[0, 0], 0., current_pos[1, 0]])
        target_state_inf = self.__call__(target_pos_inf, obstacle_center_inf, current_pos_inf)

        ### Only return pos and vel in 2D coordinates
        return np.mat(np.array([target_state_inf[[0, 2, 3, 5]]])).reshape(-1, 1)

    def fcn_det(self, test_pt, center, target):
        d_center = np.sqrt(np.sum((test_pt - center)**2))
        d_target = np.sqrt(np.sum((test_pt - target)**2))
        if d_center < d_target:
            return True
        else:
            return False

    def ccw_fcn(self, pos_test, pos_ref):
        theta1 = np.angle(pos_test[0] + 1j * pos_test[2])
        theta2 = np.angle(pos_ref[0]+ 1j*pos_ref[2])

        ### Fix quadrant issues: 
        if pos_ref[0] < 0 and pos_test[0] < 0:
            if pos_ref[2] < 0 and pos_test[2] >0:
                theta2 += 2*np.pi
            elif pos_ref[2] > 0 and pos_test[2] < 0:
                theta1 += 2*np.pi
        return theta1 > theta2

    def test_fcns(self, target_rad = 10., nreps = 100): 
        ''' make sure this is working properly
        '''
        rad = np.linspace(0., 2*np.pi, 9.)[:-1]

        for ir, r in enumerate(rad):
            # Setup a figure; 
            f, ax = plt.subplots()

            # target location: 
            target_pos = np.array([target_rad*np.cos(r), target_rad*np.sin(r)])
            obs_pos = target_pos / 2.

            ax = self.plot_patches(ax, target_pos, obs_pos)

            for n in range(nreps):
                # Sample from the workspace: 
                x = np.random.randn()*5 + target_pos[0]
                y = np.random.randn()*5 + target_pos[1]
                cp = np.array([x, y]).reshape(-1, 1)
                targ = self.call_from_sims(target_pos.reshape(-1, 1), obs_pos.reshape(-1, 1), cp)
                ax.plot([x, targ[0]], [y, targ[1]], '.-', alpha = .3)

            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
            ax.plot(target_pos[0], target_pos[1], 'r*')
            ax.set_title('Target: '+str(ir))
            f.tight_layout()

    def plot_patches(self, ax, target_pos, obs_pos):
        circle = mpatches.Circle([target_pos[0], target_pos[1]], 1.7, ec="none")
        rect = mpatches.Rectangle([obs_pos[0] - 1.5, obs_pos[1] - 1.5], 3, 3)
        patches = [circle, rect]; 

        obs_complex = np.complex(obs_pos[0], obs_pos[1])
        obs_ang = np.angle(obs_complex)
        obs_r = np.abs(obs_complex)

        for ta in [np.pi/2., -1*np.pi/2.]:
            pos = obs_pos + 6.*(np.array([np.cos(ta+obs_ang), np.sin(ta+obs_ang)]))
            circle = mpatches.Circle([pos[0], pos[1]], 0.5, ec="none")
            patches.append(circle)

        colors = np.linspace(0, 1, len(patches))
        collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
        ax.axis('equal')

        return ax
