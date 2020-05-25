import analysis_config
from online_analysis import util_fcns
from matplotlib import cm
import copy
import matplotlib as mpl
import numpy as np 
import matplotlib.pyplot as plt

def plot_pred(command_bins, push, trg, tsk, bin_num, mag_bound, targ0 = 0., targ1 = 1., min_obs = 20, min_obs2 = 5,
             arrow_scale = .004, lims = 5, prefix = '', save = False, save_dir = None, pred_push = None, scale_rad = 2.): 
    
    ### Get the mean angle / magnitudes ###
    mag_bound = scale_rad*np.hstack((mag_bound))
    ang_mns = np.linspace(0, 2*np.pi, 9)
    mag_mns = [np.mean([0., mag_bound[0]])]
    for i in range(2):
        mag_mns.append(np.mean([mag_bound[i], mag_bound[i+1]]))
    mag_mns.append(np.mean([mag_bound[2], mag_bound[2] + 2]))
    
    ### INDEX LEVEL 1 -- Get all the trg0 / trg1 data; 
    ix0 = np.nonzero(np.logical_and(trg == targ0, tsk == 0))[0]
    ix1 = np.nonzero(np.logical_and(trg == targ1, tsk == 1))[0]
    
    ### For the CO task go through commands and get the ones with enough observations ;
    com0 = command_bins[ix0, :]
    com1 = command_bins[ix1, :]
    
    psh0 = push[ix0, :]
    psh1 = push[ix1, :]
    
    if pred_push is None:
        pred_psh0 = None
        pred_psh1 = None
    else:
        pred_psh0 = pred_push[ix0, :]
        pred_psh1 = pred_push[ix1, :]
    
    bn0 = bin_num[ix0]
    bn1 = bin_num[ix1]
    
    trg0 = trg[ix0]
    trg1 = trg[ix1]
    
    
    ### Go through -- which bins have enough min_obs in both tasks; 
    keep = []
    for m in range(4):
        for a in range(8):
            
            ### INDEX LEVEL 2 -- Indices to get the mag / angle from each task 
            i0 = np.nonzero(np.logical_and(com0[:, 0] == m, com0[:, 1] == a))[0]
            i1 = np.nonzero(np.logical_and(com1[:, 0] == m, com1[:, 1] == a))[0]
            
            ### Save these ###
            if np.logical_and(len(i0) >= min_obs, len(i1) >= min_obs):
                
                ### Add the m / a / indices for this so we can go through them later; 
                keep.append([m, a, i0.copy(), i1.copy()])
    
    ### Now for each for these, plot the main thing; 
    for i_m, (m, a, i0, i1) in enumerate(keep):
        
        ### Make a figure centered on this command; 
        fig, ax_all = plt.subplots(ncols = 2, figsize = (10, 50/8.))
        fig.subplots_adjust(bottom=0.3)
        for ia, (ax, tgi) in enumerate(zip(ax_all, [targ0, targ1])):
            
            ### Set the axis to square so the arrows show up correctly 
            ax.axis('equal')
            title_string = 'cotg%d_obstg%d_mag%d_ang%d' %(targ0, targ1, m, a)
            
            ax.set_title('Task %d, Targ %d Mag %d, Ang %d' %(ia, tgi, m, a))

            ### Plot the division lines; 
            for A in np.linspace(-np.pi/8., (2*np.pi) - np.pi/8., 9):
                ax.plot([0, lims*np.cos(A)], [0, lims*np.sin(A)], 'k-', linewidth=.5)
            
            ### Plot the circles 
            t = np.arange(0, np.pi * 2.0, 0.01)
            tmp = np.hstack(([0], mag_bound, [mag_bound[2] + 2]))
            for mm in tmp:
                x = mm * np.cos(t)
                y = mm * np.sin(t)
                ax.plot(x, y, 'k-', linewidth = .5)
        
        ### Get mean neural push for each task ###
        mn0 = np.mean(psh0[i0, :], axis=0)
        mn1 = np.mean(psh1[i1, :], axis=0)
        
        ### Plot this mean neural push as big black arrow ###
        ax_all[0].quiver(mag_mns[m]*np.cos(ang_mns[a]), mag_mns[m]*np.sin(ang_mns[a]), mn0[0], mn0[1], 
                  width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')
        
        ax_all[1].quiver(mag_mns[m]*np.cos(ang_mns[a]), mag_mns[m]*np.sin(ang_mns[a]), mn1[0], mn1[1], 
                  width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')             
        
        ### ADJUSTMENT TO LEVEL 2 -- Now get the NEXT action commands ###
        i0p1 = i0 + 1; 
        i1p1 = i1 + 1; 
        
        ### If exceeds length after adding "1"
        if i0p1[-1] > (len(bn0) -1):
            i0p1 = i0p1[:-1]
        if i1p1[-1] > (len(bn1) -1):
            i1p1 = i1p1[:-1]
        ### Remove the commands that are equal to the minimum bin_cnt (=1) 
        ### because they spilled over from the last trial ###
        kp0 = np.nonzero(bn0[i0p1] > 1)[0]
        kp1 = np.nonzero(bn1[i1p1] > 1)[0]
        
        ### Keep these guys ###
        i0p1 = i0p1[kp0]
        i1p1 = i1p1[kp1]
        
        ### Arrows dict 
        ### We need to create this because we're not sure how big to maek the colormap when plotting; 
        arrows_dict = {}
        arrows_dict[0] = []
        arrows_dict[1] = []
        
        pred_arrows_dict = {}
        pred_arrows_dict[0] = []
        pred_arrows_dict[1] = []
        
        index_dict = {}
        index_dict[0] = []
        index_dict[1] = []
        
        ### These should all be the same target still ###
        assert(np.all(trg0[i0p1] == targ0))
        assert(np.all(trg1[i1p1] == targ1))
        
        ### Iterate through the tasks ###

        for m in range(4):
            for a in range(8):
                ### LEVEL 3 -- OF THE NEXT ACTION COMMANDS, which match the action #####
                j0 = np.nonzero(np.logical_and(com0[i0p1, 0] == m, com0[i0p1, 1] == a))[0]
                j1 = np.nonzero(np.logical_and(com1[i1p1, 0] == m, com1[i1p1, 1] == a))[0]

                for i_, (index, index_og, pshi, predpshi) in enumerate(zip([j0, j1], [i0p1, i1p1], [psh0, psh1],
                                                                [pred_psh0, pred_psh1])):
                    if len(index) >= min_obs2:
                        print('Adding followup tsk %d, m %d, a %d' %(i_, m, a))

                        ### Great plot this;
                        mn_next_action = np.mean(pshi[index_og[index], :], axis=0)
                        xi = mag_mns[m]*np.cos(ang_mns[a])
                        yi = mag_mns[m]*np.sin(ang_mns[a])
                        vx = mn_next_action[0];
                        vy = mn_next_action[1]; 

                        arrows_dict[i_].append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(vx), copy.deepcopy(vy), len(index)])
                        index_dict[i_].append(len(index))
                        
                        #### Predicted Action ####
                        if predpshi is None:
                            pass
                        else:
                            pred_mn_next_action = np.mean(predpshi[index_og[index], :], axis=0)
                            p_vx = pred_mn_next_action[0]
                            p_vy = pred_mn_next_action[1]
                            
                            pred_arrows_dict[i_].append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(p_vx), copy.deepcopy(p_vy)])
        
        #import pdb; pdb.set_trace()
        ### Now figure out the color lists for CO / OBS separately; 
        mx_co = np.max(np.hstack((index_dict[0])))
        mn_co = np.min(np.hstack((index_dict[0])))
        co_cols = np.linspace(0., 1., mx_co - mn_co + 1)
        co_colors = [cm.viridis(x) for x in co_cols]
        print('Co mx = %d, mn = %d' %(mx_co, mn_co))
        
        mx_ob = np.max(np.hstack((index_dict[1])))
        mn_ob = np.min(np.hstack((index_dict[1])))
        obs_cols = np.linspace(0., 1., mx_ob - mn_ob+1)
        obs_colors = [cm.viridis(x) for x in obs_cols]
        print('Obs mx = %d, mn = %d' %(mx_ob, mn_ob))
        
        for tsk, (mnN, cols) in enumerate(zip([mn_co, mn_ob], [co_colors, obs_colors])):
            
            print('Len arrows_dict[%d]: %d' %(tsk, len(arrows_dict[tsk])))
            ### go through all the arrows;
            for arrow in arrows_dict[tsk]:
                
                ### Parse the parts 
                xi, yi, vx, vy, N = arrow; 
                
                ### Plot it;
                ax_all[tsk].quiver(xi, yi, vx, vy,
                                  width = arrow_scale, color = cols[N - mnN], 
                                    angles='xy', scale=1, scale_units='xy')
            
            for pred_arrow in pred_arrows_dict[tsk]:
                
                ### Parse the parts
                pxi, pyi, pvx, pvy = pred_arrow
                
                ### Plot it; 
                ax_all[tsk].quiver(pxi, pyi, pvx, pvy, 
                                  angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                  linestyle = 'dashed', edgecolor='r', facecolor='none', 
                                  linewidth = 1.)

        for ia, ax in enumerate(ax_all):
            ax.set_xlim([-lims, lims])
            ax.set_ylim([-lims, lims])
            
        ### Set colorbar; 
        cmap = mpl.cm.viridis
        if mn_co == mx_co:
            norm = mpl.colors.Normalize(vmin=mn_co, vmax=mx_co+.1)
        else:
            norm = mpl.colors.Normalize(vmin=mn_co, vmax=mx_co)
        cax0 = fig.add_axes([0.1, 0.1, 0.3, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(cax0, cmap=cmap,
                            norm=norm, orientation='horizontal',
                            boundaries = np.arange(mn_co-0.5, mx_co+1.5, 1.))
        cb1.set_ticks(np.arange(mn_co, mx_co+1))
        cb1.set_ticklabels(np.arange(mn_co, mx_co+1))

            
        
        cb1.set_label('Counts')
        if mn_ob == mn_ob:
            norm = mpl.colors.Normalize(vmin=mn_ob, vmax=mx_ob+.1)
        else:
            norm = mpl.colors.Normalize(vmin=mn_ob, vmax=mx_ob)
        cax1 = fig.add_axes([0.6, .1, .3, .05])
        cb1 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap,
                            norm=norm, orientation='horizontal',
                            boundaries = np.arange(mn_ob-0.5, mx_ob+1.5, 1.))
        cb1.set_label('Counts')
        cb1.set_ticks(np.arange(mn_ob, mx_ob+1))
        cb1.set_ticklabels(np.arange(mn_ob, mx_ob+1)) 
        
        
        ### Save this ? 
        if save:
            fig.savefig(save_dir+'/'+prefix+'_'+title_string+'.png')

def plot_pred_across_full_day(command_bins, push, bin_num, mag_bound, min_obs2 = 20,
             arrow_scale = .004, lims = 5, prefix = '', save = False, save_dir = None, 
             pred_push = None, scale_rad = 2.): 
    '''
    command_bins -- discretized commands over all bins 
    push -- continuous pushses
    bin_num -- indices within the trial 
    mag_bound -- magnitude boundaries
    min_obs2 -- number of counts of future actions needed to plot; 

    pred_push -- general dynamics predictions; 
    scale_rad -- how to scale out the radius for better visualization; 
    '''

    ### Get the mean angle / magnitudes ###
    mag_bound = scale_rad*np.hstack((mag_bound))
    ang_mns = np.linspace(0, 2*np.pi, 9)
    mag_mns = [np.mean([0., mag_bound[0]])]
    for i in range(2):
        mag_mns.append(np.mean([mag_bound[i], mag_bound[i+1]]))
    mag_mns.append(np.mean([mag_bound[2], mag_bound[2] + 2]))
    
    ### Go through -- which bins have enough min_obs in both tasks; 
    keep = []
    for m in range(4):
        for a in range(8):
            
            ### INDEX LEVEL 2 -- Indices to get the mag / angle from each task 
            ix = np.nonzero(np.logical_and(command_bins[:, 0] == m, command_bins[:, 1] == a))[0]
            
            ### Add the m / a / indices for this so we can go through them later; 
            keep.append([m, a, ix.copy()])
    
    ### Now for each for these, plot the main thing; 
    for i_m, (m, a, ixi) in enumerate(keep):
        
        ### Make a figure centered on this command; 
        fig, ax_all = plt.subplots(figsize = (8, 10))
        fig.subplots_adjust(bottom=0.3)
        
        ### Set the axis to square so the arrows show up correctly 
        ax_all.axis('equal')
        title_string = 'ALL_mag%d_ang%d' %(m, a)

        ### Plot the division lines; 
        for A in np.linspace(-np.pi/8., (2*np.pi) - np.pi/8., 9):
            ax_all.plot([0, lims*np.cos(A)], [0, lims*np.sin(A)], 'k-', linewidth=.5)
            
        ### Plot the circles 
        t = np.arange(0, np.pi * 2.0, 0.01)
        tmp = np.hstack(([0], mag_bound, [mag_bound[2] + 2]))
        for mm in tmp:
            x = mm * np.cos(t)
            y = mm * np.sin(t)
            ax_all.plot(x, y, 'k-', linewidth = .5)
        
        ### Get mean neural push for each task ###
        segment_mn = np.mean(push[ixi, :], axis=0)
        
        ### Plot this mean neural push as big black arrow ###
        ax_all.quiver(mag_mns[m]*np.cos(ang_mns[a]), mag_mns[m]*np.sin(ang_mns[a]), segment_mn[0], segment_mn[1], 
                  width=arrow_scale*2, color = 'k', angles='xy', scale=1, scale_units='xy')
        
        ### ADJUSTMENT TO LEVEL 2 -- Now get the NEXT action commands ###
        ixip1 = ixi + 1; 
        
        ### If exceeds length after adding "1"
        if len(ixi) > 0:
            if ixip1[-1] > (len(push) -1):
                ixip1 = ixip1[:-1]
       
            ### Remove the commands that are equal to the minimum bin_cnt (=1) 
            ### because they spilled over from the last trial ###
            kp0 = np.nonzero(bin_num[ixip1] > 1)[0]
            
            ### Keep these guys ###
            index_og = ixip1[kp0]
            
            ### Arrows dict 
            ### We need to create this because we're not sure how big to maek the colormap when plotting; 
            arrows_dict = []        
            pred_arrows_dict = []
            index_dict = []
            
            ### Iterate through the tasks ###
            for mp in range(4):
                for ap in range(8):

                    ### LEVEL 2 -- OF THE NEXT ACTION COMMANDS, which match the action #####
                    index = np.nonzero(np.logical_and(command_bins[index_og, 0] == mp, command_bins[index_og, 1] == ap))[0]

                    if len(index) >= min_obs2:
                        print('Followup, m %d, a %d' %(mp, ap))
                        
                        ### Great plot this;
                        mn_next_action = np.mean(push[index_og[index], :], axis=0)
                        xi = mag_mns[mp]*np.cos(ang_mns[ap])
                        yi = mag_mns[mp]*np.sin(ang_mns[ap])
                        vx = mn_next_action[0];
                        vy = mn_next_action[1]; 

                        arrows_dict.append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(vx), copy.deepcopy(vy), len(index)])
                        index_dict.append(len(index))
                        
                        #### Predicted Action ####
                        if pred_push is None:
                            pass
                        else:
                            pred_mn_next_action = np.mean(pred_push[index_og[index], :], axis=0)
                            p_vx = pred_mn_next_action[0]
                            p_vy = pred_mn_next_action[1]
                            
                            pred_arrows_dict.append([copy.deepcopy(xi), copy.deepcopy(yi), 
                                                 copy.deepcopy(p_vx), copy.deepcopy(p_vy), mp, ap])
            
            #import pdb; pdb.set_trace()
            ### Now figure out the color lists for CO / OBS separately; 
            if len(index_dict) > 0:
                num_corr = [0, 0] ## Total correct / total assessed; 

                mx_ = np.max(np.hstack((index_dict)))
                mn_ = np.min(np.hstack((index_dict)))
                cols = np.linspace(0., 1., mx_ - mn_ + 1)
                colors = [cm.viridis(x) for x in cols]
                print('All mx = %d, mn = %d' %(mx_, mn_))
            
                for arrow in arrows_dict:
                    
                    ### Parse the parts 
                    xi, yi, vx, vy, N = arrow; 
                    
                    ### Plot it;
                    ax_all.quiver(xi, yi, vx, vy,
                                      width = arrow_scale, color = colors[N - mn_], 
                                        angles='xy', scale=1, scale_units='xy')
                
                for pred_arrow in pred_arrows_dict:
                    
                    ### Parse the parts
                    pxi, pyi, pvx, pvy, mi, ai = pred_arrow
                    
                    ### Plot the arrows in "outline / dashd red" if they are correct
                    ### Plot them in "filled in alpha red" if they are wrong; 
                    corr_dir = assess_corr_dir([m, a], [mi, ai], segment_mn, np.array([pvx, pvy]))

                    if corr_dir is None: 
                        ax_all.quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='gray', facecolor='none', 
                                          linewidth = 1.)

                    elif corr_dir == True: 
                        ### Plot it; 
                        num_corr[0] += 1
                        num_corr[1] += 1
                        
                        ax_all.quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          linestyle = 'dashed', edgecolor='r', facecolor='none', 
                                          linewidth = 1.)


                    elif corr_dir == False: 
                        num_corr[1] += 1
                        ### Plot it; 
                        ax_all.quiver(pxi, pyi, pvx, pvy, 
                                          angles = 'xy', scale_units = 'xy', scale = 1, width=arrow_scale,
                                          facecolor='r', alpha = 0.4, linewidth=0.)

                
            
                ax_all.set_xlim([-lims, lims])
                ax_all.set_ylim([-lims, lims])

                pc = int(100*(float(num_corr[0])/float(num_corr[1])))
                ax_all.set_title('Mag %d, Ang %d, Perc Corr: %d' %( m, a, pc))

                ### Set colorbar; 
                cmap = mpl.cm.viridis
                if mn_ == mx_:
                    norm = mpl.colors.Normalize(vmin=mn_, vmax=mx_+.1)
                else:
                    norm = mpl.colors.Normalize(vmin=mn_, vmax=mx_)
                cax0 = fig.add_axes([0.15, 0.1, 0.7, 0.05])
                cb1 = mpl.colorbar.ColorbarBase(cax0, cmap=cmap,
                                    norm=norm, orientation='horizontal',
                                    boundaries = np.arange(mn_-0.5, mx_+1.5, 1.))
                tmp2 = np.mean([mn_, mx_+1])
                cb1.set_ticks([mn_, tmp2, mx_+1]) 
                cb1.set_ticklabels([mn_, tmp2, mx_+1])
                cb1.set_label('Counts')

                ### Save this ? 
                if save:
                    fig.savefig(save_dir+'/'+prefix+'_'+title_string+'.png')

def assess_corr_dir(disc_dir, disc_pred_dir, mn_dir, mn_pred_dir):
    m, a = disc_dir
    mi, ai = disc_pred_dir

    if a ==  ai:
        return None
    else:

        ang_mns = np.linspace(0, 2*np.pi, 9)

        ### Center angle ####
        disc_ang = np.array([np.cos(ang_mns[a]), np.sin(ang_mns[a])])

        ### Predicted angle ###
        disc_ang_pred = np.array([np.cos(ang_mns[ai]), np.sin(ang_mns[ai])])

        ## Ok now do the cross product to see if CW (-1) or CCW (+1)
        true_cp = np.sign(np.cross(disc_ang, disc_ang_pred))

        ### Now do the predicted 
        act_cp = np.sign(np.cross(mn_dir, mn_pred_dir))

        return true_cp == act_cp

