import feedback_controllers
import numpy as np
import matplotlib.pyplot as plt

def simple_A(offs = [0, 0], alpha = .7, dt = .1):

    ### pos/vel/offset A matrix; 
    A = np.mat([[1, 0, dt, 0, dt*offs[0]], 
                [0, 1, 0, dt, dt*offs[1]],
                [0, 0, alpha, 0, offs[0]], 
                [0, 0, 0, alpha, offs[1]],
                [0, 0, 0, 0,      1]])

    ### 2D input drives vx,vy and instantaneously updates pos w/ integrated vel; 
    B = np.mat(np.zeros((5, 2)))
    B[0, 0] = dt*1.
    B[1, 1] = dt*1.
    B[2, 0] = 1.
    B[3, 1] = 1.

    #### Setup LQR: 
    # Cost on the state 
    Q = np.mat(np.eye(5))

    ### Cost on the input; 2 x 2 matrix 
    R = np.mat(np.eye(2))

    # LQR controller: 
    lqr = feedback_controllers.LQRController(A, 
        B, Q, R)

    ### State feedback matrix, infinite horizon
    K = lqr.F
    
    ### angles for different targets 
    ang = np.linspace(0, 2*np.pi, 9)[:-1]
    rad = 10.

    f, ax = plt.subplots()

    for ia, ang_i in enumerate(ang):
        current_state = np.mat(np.array([0., 0., 0., 0., 1.])[:, np.newaxis])

        #### Targets state -- position w/ zero velocity; 
        target_state = np.mat(np.array([rad*np.cos(ang_i), rad*np.sin(ang_i), 0, 0, 1.])[:, np.newaxis])

        state_hist = [current_state]

        #### 
        for i_t in range(1000):
            state_err = current_state - target_state

            ### Change the offset term to 1; https://www.cc.gatech.edu/~bboots3/ACRL-Spring2019/Lectures/LQR_notes.pdf, page 31; 
            state_err[-1] = 1; 

            # Get input via state feedback 
            u = -1*np.dot(K, state_err)

            # Update state with feedback 
            current_state = np.dot(A, current_state) + np.dot(B, u)
            state_hist.append(current_state)

            if in_target(current_state, target_state):
                print('Reached target %d, N steps = %d' %(ia, i_t))
                state_hist = np.array(np.hstack((state_hist)))
                ax.plot(state_hist[0, :], state_hist[1, :], 'k.-')
                break

            if i_t == 999:
                print('Failed to reach') 
                state_hist = np.array(np.hstack((state_hist)))
                ax.plot(state_hist[0, :], state_hist[1, :], 'r.-')


def in_target(state, target_state):
    if np.linalg.norm(state[:2] - target_state[:2]) < 1:
        return True
    else:
        return False