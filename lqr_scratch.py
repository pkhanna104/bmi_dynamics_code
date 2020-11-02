def sim_lqr_nk_co_trial_finite_horizon(A,B,K,T,target,state_init, state_label, input_label, num_neurons, max_iter=1e5, hold_req=2, target_r=1.7):
    """
    simulates lqr control of a joint neural-kinematic dynamical system for a center-out trial 
    (moving straight from start state to target state)
    assumes: TODO
    in state vector, first all neurons are listed, then all cursor kinematics are listed.
    """

    #Target state
    state_T = target
    state_dim = state_T.shape[0]
    #steady input for target state:
    u_star = 0#np.linalg.pinv(B)*(np.eye(state_dim)-A)*state_T
    u_T = 0# -np.linalg.pinv(B)*A*state_T

    # print(u_T)
    # u_star = B_inv*np.mat(np.eye(state_dim)-A)*state_T

    #State
    state = copy.copy(state_init)
    state_list = [state]
    #State error
    state_e = state-state_T
    state_e_list = [state_e]
    #Input
    u_list = []

    #Simulate trial: 
    trial_complete = False
    sim_len = 1
    hold_i = 0
    
    for t in range(0,T-1):
        #Calculate u
        # u_T = np.linalg.pinv(np.linalg.pinv((np.eye(state_dim)-(A+B*K[0,t])))*B)*state_T
        # u = K[0,t]*state_e + u_T

        # u_T = np.linalg.pinv(B)*(np.eye(state_dim)-(A+B*K[0,t]))*state_T
        u = K[0,t]*state_e
        # u = K[0,t]*state_e
        u_list.append(u)

        #Calculate state_e
        state_e = A*state_e + B*u
        state_e_list.append(state_e)

        #Calculate state: 
        state = state_e + state_T
        state_list.append(state)

        if not trial_complete:
            sim_len+=1
            dist2target = np.linalg.norm(state_e[num_neurons:(num_neurons+2)])
            if dist2target <= target_r:
                hold_i+=1
            else:
                hold_i=0
            if(hold_i>=hold_req):
                trial_complete = True
            
    #RESULTS:

    #state error:
    state_e_mat = np.array(state_e_list).squeeze().T
    state_e_da = xr.DataArray(state_e_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs'])

    #input:
    u_mat = np.array(u_list).squeeze().T
    u_da = xr.DataArray(u_mat, coords={'v':input_label,'obs':np.arange(T-1)}, dims=['v', 'obs'])

    #state:
    state_mat = np.array(state_list).squeeze().T
    # state_mat[num_neurons:,:] = state_mat[num_neurons:,:] + state_T[num_neurons:,:] #add the target kinematic state back to the error
    state_da = xr.DataArray(state_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs']) 

    return u_da, state_da, state_e_da, sim_len    






#Attempt from 10/30/2020 at 2:42pm
# def sim_lqr_nk_co_trial_finite_horizon(A,B,K,T,target,state_init, state_label, input_label, num_neurons, max_iter=1e5, hold_req=2, target_r=1.7):
#     """
#     simulates lqr control of a joint neural-kinematic dynamical system for a center-out trial 
#     (moving straight from start state to target state)
#     assumes: TODO
#     in state vector, first all neurons are listed, then all cursor kinematics are listed.
#     """

#     #Target state
#     state_T = target
#     #e_sub = np.linalg.pinv(A)*state_T

#     #State
#     state = copy.copy(state_init)
#     state_list = [state]
#     #State error
#     state_e = state-state_T
#     state_e_list = [state_e]
#     #Input
#     u_list = []

#     #Simulate trial: 
#     trial_complete = False
#     sim_len = 1
#     hold_i = 0
    
#     for t in range(0,T-1):
#         #Calculate u
#         u = K[0,t]*state_e
#         # u = K[0,t]*(state-e_sub)
#         u_list.append(u)

#         #Calculate state
#         state = A*state+B*u
#         state_list.append(state)

#         #Calculate state_e
#         state_e = state-state_T
#         state_e_list.append(state_e)

#         if not trial_complete:
#             sim_len+=1
#             dist2target = np.linalg.norm(state_e[num_neurons:(num_neurons+2)])
#             if dist2target <= target_r:
#                 hold_i+=1
#             else:
#                 hold_i=0
#             if(hold_i>=hold_req):
#                 trial_complete = True
            
#     #RESULTS:

#     #state error:
#     state_e_mat = np.array(state_e_list).squeeze().T
#     state_e_da = xr.DataArray(state_e_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs'])

#     #input:
#     u_mat = np.array(u_list).squeeze().T
#     u_da = xr.DataArray(u_mat, coords={'v':input_label,'obs':np.arange(T-1)}, dims=['v', 'obs'])

#     #state:
#     state_mat = np.array(state_list).squeeze().T
#     # state_mat[num_neurons:,:] = state_mat[num_neurons:,:] + state_T[num_neurons:,:] #add the target kinematic state back to the error
#     state_da = xr.DataArray(state_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs']) 

#     return u_da, state_da, state_e_da, sim_len


# 10/30/2020 9:38am: I think there's a bug with this, cuz I propagated state error, not state:
# def sim_lqr_nk_co_trial_finite_horizon(A,B,K,T,target,state_init, state_label, input_label, num_neurons, max_iter=1e5, hold_req=2, target_r=1.7):
#     """
#     simulates lqr control of a joint neural-kinematic dynamical system for a center-out trial 
#     (moving straight from start state to target state)
#     assumes: TODO
#     in state vector, first all neurons are listed, then all cursor kinematics are listed.
#     """

#     state_T = target
#     state_e_init = state_init-state_T
#     state_e_list = []
#     state_e = state_e_init
#     state_e_list.append(state_e)

#     #Input
#     u_list = []

#     #Simulate trial: 
#     trial_complete = False
#     sim_len = 1
#     hold_i = 0

#     for t in range(0,T-1):
#         state_e = (A+B*K[0,t])*state_e
#         state_e_list.append(state_e)
#         u = K[0,t]*state_e
#         u_list.append(u)
#         if not trial_complete:
#             sim_len+=1
#             dist2target = np.linalg.norm(state_e[num_neurons:(num_neurons+2)])
#             if dist2target <= target_r:
#                 hold_i+=1
#             else:
#                 hold_i=0
#             if(hold_i>=hold_req):
#                 trial_complete = True
            
#     #RESULTS:
#     #input:
#     u_mat = np.array(u_list).squeeze().T
#     u_da = xr.DataArray(u_mat, coords={'v':input_label,'obs':np.arange(T-1)}, dims=['v', 'obs'])

#     #state error:
#     state_e_mat = np.array(state_e_list).squeeze().T
#     # state_e_mat = state_e_mat.squeeze().T
#     state_e_da = xr.DataArray(state_e_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs'])

#     #state:
#     state_mat = copy.deepcopy(state_e_mat)
#     state_mat[num_neurons:,:] = state_mat[num_neurons:,:] + state_T[num_neurons:,:] #add the target kinematic state back to the error
#     state_da = xr.DataArray(state_mat, coords={'v':state_label,'obs':np.arange(T)}, dims=['v', 'obs']) 

#     return u_da, state_da, state_e_da, sim_len  
