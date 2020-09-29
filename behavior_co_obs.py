import scipy.io as sio
import scipy.stats as sio_stat
import scipy.interpolate
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import pickle
import sys
# sys.path.append('/Library/Python/2.7/site-packages')
# sys.path.append('/Library/Python/2.7/bin')
# import imp
# from mpldatacursor import datacursor

from preeya_co_obstacle import data_for_v
# import imp
# preeyacode_file = '/Users/vivekathalye/Dropbox/Code/preeya_co_obstacle/data_for_v.py'
# preeyacode = imp.load_source('preeyacode', preeyacode_file)

import time
import pylab as pl
from IPython import display
import copy
from bmi_dynamics_code import util as bmi_util

#TODO: function which decides which way the trajectory goes around the obstacle
# def obs_traj_cw_vs_ccw(traj_x, traj_y, target_pos):
def traj_signed_area_about_target_axis(traj_x, traj_y, target_pos):
    """
    computes the signed area of a trajectory around a target axis.  
    the target orthogonal axis is a counterclockwise rotation of 90 deg
    0) construct target axis (t) + target orthog axis (to)
    1) project data on target axis (t_proj) and target orthog axis (to_proj)
    2) compute riemann sum as: np.sum(np.diff(t_proj)*to_proj[1:])
    """
    #0)
    t,to = target_axes(target_pos)

    #1)
    traj = np.vstack((traj_x, traj_y))
    t_proj = np.dot(t, traj)
    to_proj = np.dot(to, traj)

    #2)
    a_traj = np.hstack((0, np.diff(t_proj)*to_proj[1:]))
    sa = np.sum(a_traj)
    a = np.sum(np.abs(a_traj))
    d_dic = {'kin_px':traj_x, 'kin_py':traj_y, 
    'kin_pt':t_proj, 'kin_pto':to_proj, 
    'kin_pto_area':a_traj, 'kin_pto_cum_area':np.cumsum(a_traj)}
    df = pd.DataFrame(data=d_dic)

    #Return a dataframe for traj, proj, area
    #Return the area as a
    return sa, a, df   

def target_axes(target_pos):
    """
    Constructs a right-handed coordinate system where the x-axis is the axis going from (0,0) to target_pos
    and the y-axis is ccw 90 deg rotation of the x-axis
    """
    x = target_pos/np.linalg.norm(target_pos, ord=2)
    y = np.dot(rot2D_mat(np.pi/2), x)
    return x,y

def rot2D_mat(theta):
    """
    returns a 2D rotation matrix, which rotates points counterclockwise by theta
    theta: angle (rad) 
    """
    mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return mat

def cartesian2polar(y,x):
    """
    Input: y, x
    Assumes y,x are 1d numpy arrays
    """
    d = np.vstack((y,x)) #2 x num_samples
    mag = np.linalg.norm(d,ord=2,axis=0)
    angle = np.arctan2(y,x)
    return mag, angle

#COMMENT: I don't think this works in general, it worked for the case I was using in notebook:
#'psth_polar_interp_separate_cw_ccw.ipynb'
# def center_angle(angle, ctr_orig, ctr_new):
#     """
#     ctr_orig: the center angle of angle data.  
#     e.g. if angle ranges from (0,2*pi) then the center is pi.  
#     if angle ranges from (-pi,pi) then the center is 0.
#     """
#     angle_center = (angle+(np.pi-ctr_orig)-ctr_new)%(2*np.pi)-(np.pi-ctr_orig)
#     return angle_center   

#COMMENT: This code is the one to use... not center_angle...
def center_angle(angle, ctr):
    """
    angle: array-like
    ctr: the center angle of angle data.  
    e.g. if you want angle to range from (0,2*pi) then the center is pi.  
    if you want angle to range from (-pi,pi) then the center is 0.
    """
    angle = np.array(angle)
    angle2round = np.array((ctr-angle)/(2*np.pi))
    angle_center = angle+np.matrix.round(angle2round)*2*np.pi
    # angle_center = angle+np.matrix.round((ctr-angle)/(2*np.pi))*2*np.pi
    return angle_center       

def bin_data_pt(data_pt, bins): 
    #assumes: 
    #bins is 2 x num_bins
    #top row is bin's bottom edge
    #bot row is bin's top edge
    bin_val = np.where((data_pt >= bins[0,:]) & (data_pt <= bins[1,:]))[0]
    return bin_val

def bin_vec_data(vec_data, bin_dic): 
    """
    input: vec_data, bin_dic
    output: bin_result, hist_result

    assumes: 
    vec_data is num_observations X num_dim

    bin_dic is a dictionary num_dim keys from 0,...,num_dim-1
    Each entry in bin_dic is a matrix of 2xnum_bins, with:
    top row is bin's bottom edge
    bot row is bin's top edge

    """
    #Loop over each dimension and bin: 
    num_dim = vec_data.shape[1]
    num_bin_over_dim = np.zeros(num_dim)
    for dim_i in np.arange(0,num_dim):
        num_bin_over_dim[dim_i] = bin_dic[dim_i].shape[1]

    bin_result = np.zeros(vec_data.shape)
    #loop each dimension and bin data
    for dim_i in np.arange(0,num_dim):
        bin_fn =lambda x: np.where((x >= bin_dic[dim_i][0,:]) & (x <= bin_dic[dim_i][1,:]))[0] 
        bin_result[:,dim_i] = np.hstack(map(bin_fn, vec_data[:,dim_i]))
    
    #loop each observation and make histogram
    # print(num_bin_over_dim)
    hist_result = np.zeros(num_bin_over_dim.astype(int))
    for i in np.arange(0,bin_result.shape[0]):
        obs_i = tuple(bin_result[i,:].astype(int))
        # print(obs_i)
        hist_result[obs_i] +=1

    return bin_result, hist_result

def remove_data_out_bound(vec_data, lim_dic):
    """
    assumes: 
    vec_data is num_observations X num_dim
    lim_dic is a dictionary num_dim keys from 0,...,num_dim-1
    each entry of lim_dic is a np array of len 2 with lower and upper limits on a dimension

    #returns:
    #for each dimension, which entries are out of bounds
    #the deleted idxs

    """
    out_bound_over_dim = {}
    del_idxs = []
    for dim in range(vec_data.shape[1]):
        out_bound_over_dim[dim] = []
        out_bound = np.where(np.logical_or(vec_data[:,dim]<lim_dic[dim][0], vec_data[:,dim]>lim_dic[dim][1]))[0]
        if len(out_bound) >0:
            out_bound_over_dim[dim] = out_bound
            del_idxs.append(out_bound)
    if len(del_idxs) > 0:
        del_idxs = np.unique(np.concatenate(del_idxs))
        vec_data = np.delete(vec_data, del_idxs, axis=0)
    else:
        del_idxs = np.array([])
    return vec_data, del_idxs, out_bound_over_dim

def plot_2d_hist(z, y_min, y_max, y_num_bins, x_min, x_max, x_num_bins, vmin, vmax, title, xlabel, ylabel, im_size, cmap='viridis'):
    y,x = np.meshgrid(np.linspace(y_min, y_max, y_num_bins), np.linspace(x_min, x_max, x_num_bins))
    fig, ax = plt.subplots(figsize=(im_size,im_size))
    c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.axis('square')
    fig.colorbar(c, ax=ax)
    plt.show()
    return fig,ax        

def store_data2_by_data1_bin(data2bin, data2store, bin_dic, lim_dic):
    """
    bin_dic: dictionary num_dim keys from 0,...,num_dim-1
    data2bin: num_observations X num_dimensions
    this data is binned using bin_dic
    data2store: this data is stored according to the bin
    lim_dic: dictionary containing the limits on data2bin.  this function throws out data which doesn't obey bounds.
    Returns a dictionary with num_dimensions, keys are the bin number
    """
    #Analyze in-bound data: 
    data2bin, del_idxs, out_bound_over_dim = remove_data_out_bound(data2bin, lim_dic)
    data2store = np.delete(data2store, del_idxs, axis=0)
    #bin data
    bin_r, hist_r = bin_vec_data(data2bin, bin_dic)
    bin_r = bin_r.astype(int)
    # #initialize r:
    # num_dim = data2bin.shape[1]
    # num_bin = np.zeros(num_dim)
    # for dim in range(num_dim):
    #     num_bin[dim] = bin_dic[dim].shape[1]
    #store data
    r = {}
    for i in range(bin_r.shape[0]):
        b = tuple(bin_r[i,:])
        d = data2store[i,:]
        if b not in r: 
            r[b] = []
        r[b].append(d)
    return r, data2bin, data2store, del_idxs, out_bound_over_dim

def bin_circular_pt(data_pt, bins): 
    #assumes: 
    #bins is 2 x num_bins
    #top row is bin's bottom edge
    #bot row is bin's top edge
    
    #compute angle diff between boundaries: 
    diff = (data_pt*np.ones(bins.shape)-bins) 
#     print(diff)
    diff[np.where(diff > np.pi)] -= 2*np.pi
    diff[np.where(diff < -np.pi)] += 2*np.pi
    
    bin_val = np.where((diff[0,:] >= 0) & (diff[1,:] <= 0))[0]
    return bin_val

def bin_angle_mag(data_2d, mag_bins, angle_bins):
    #data: num_observations x 2
    #mag_bins, angle_bins: 
    num_observations = data_2d.shape[0]
    data_mag = np.linalg.norm(data_2d, axis=1)
    data_angle = np.arctan2(data_2d[:,1], data_2d[:,0])
#     print(data_angle.shape)
    
    mag_bin_data = []
    angle_bin_data = []
    
    num_mag_bins = mag_bins.shape[0]
    num_angle_bins = angle_bins.shape[0]
    mag_hist = np.zeros(mag_bins.shape[1])
    angle_hist = np.zeros(angle_bins.shape[1])
    joint_hist = np.zeros((angle_bins.shape[1], mag_bins.shape[1]))
    
    for obs_i in np.arange(0,num_observations):
        mag_i = data_mag[obs_i]
#         print(mag_i)
        mag_bin = bin_data_pt(mag_i, mag_bins)
        mag_bin_data.append(mag_bin)
        mag_hist[mag_bin] +=1
        
        angle_i = data_angle[obs_i]
#         print(angle_i*180/np.pi)
        angle_bin = bin_circular_pt(angle_i, angle_bins)
        angle_bin_data.append(angle_bin)
        angle_hist[angle_bin] +=1
        
        joint_hist[angle_bin, mag_bin] +=1

#     print('mag_bin_data:')
#     print(mag_bin_data)        
#     print('angle_bin_data:')
#     print(angle_bin_data)
    bin_result = np.vstack((np.hstack(mag_bin_data), np.hstack(angle_bin_data))).T
    return bin_result, joint_hist, angle_hist, mag_hist, data_angle, data_mag

def plot_data_and_polar_bins(cartesian_data, mag_bin_edges, angle_bin_edges):
    #assumes: data is cartesian, num_observations x 2,
    num_mag_bins = mag_bin_edges.shape[1]
    num_angle_bins = angle_bin_edges.shape[1]

    #Plot bins: 
    im_size = 2
    f, ax = plt.subplots(1,1,figsize=(im_size,im_size))

    #Plot circles of different radii: 
    #how many theta's to plot: 
    num_theta = 100
    circle_theta = np.linspace(0,2*np.pi,num_theta)
    for bin_i in np.arange(0,num_mag_bins):
        for i in np.arange(0,num_theta-1): 
            mag_i = mag_bin_edges[0,bin_i]
            x = mag_i*np.array([np.cos(circle_theta[i]), np.cos(circle_theta[i+1])])
            y = mag_i*np.array([np.sin(circle_theta[i]), np.sin(circle_theta[i+1])])        
            ax.plot(x,y,color='gray')

    #Plot angle bins:

    #only plotting up to second to last magnitude bin
    max_mag = mag_bin_edges[1,-2]
    for bin_i in np.arange(0,num_angle_bins):
        x = max_mag*np.array([0, np.cos(angle_bin_edges[0,bin_i])])
        y = max_mag*np.array([0, np.sin(angle_bin_edges[0,bin_i])])
        ax.plot(x,y)

    #Plot data: 
    num_data = cartesian_data.shape[0]
    for i in np.arange(0,num_data):
        ax.plot([0, cartesian_data[i,0]], [0,cartesian_data[0,1]], color='black')     
    return f, ax    

def polar_heat_map(heat_data, mag_bin, angle_bin, cmap, vmin, vmax):
    rad = mag_bin
    azm = angle_bin

    r, th = np.meshgrid(rad, azm)
    z = heat_data

    im_size = 6
    fig = plt.figure()
    fig.set_size_inches(im_size, im_size)
    ax = Axes3D(fig)
    plt.subplot(projection="polar")

    c = plt.pcolormesh(th, r, z, cmap='Greys', vmin=vmin, vmax=vmax)
    plt.plot(azm, r, color='k', ls='none') 
    plt.thetagrids([(theta * 45)+45.0/2 for theta in range(360//45)])
    plt.rgrids(mag_bin)
    plt.grid()
    plt.colorbar(c)
    plt.show() 

    return fig, ax

def plot_polar_bins(angle_bin_edges, mag_bin_edges, angle_bin_colors, angle_linewidth=3):
    """
    8.20.2020
    INPUT: 
    bin_edges (for angle and mag): 2xnum_bins.  top row = lower bin edge, bot row = upper bin edge
    angle_colors = colors for each angle bin
    """
    mag_max = mag_bin_edges[-1,-1]
    angle_bin_c = np.mean(angle_bin_edges,axis=0)
    mag_bin = np.hstack((mag_bin_edges[0,:], mag_max))
    for i,b in enumerate(angle_bin_edges.T):
        x1 = np.cos(b[0])*mag_max
        y1 = np.sin(b[0])*mag_max
#         plt.plot([0, x1], [0, y1], color=target_color[i])
        plt.plot([0, x1], [0, y1], '--', color=np.ones(3)*0.8,zorder=0)#'k')  
    
    for i,a in enumerate(angle_bin_c): 
        x1 = np.cos(a)*mag_max
        y1 = np.sin(a)*mag_max
        plt.plot([0, x1], [0, y1], color=angle_bin_colors[i], linewidth=angle_linewidth,zorder=0)

    for b in mag_bin:
        theta = np.linspace(0,2*np.pi,1000)
        plt.plot(b*np.cos(theta), b*np.sin(theta), 'k',zorder=0)
    plt.axis('square')
    plt.xlabel('x')
    plt.ylabel('y')        

def calc_command_trials_dic_da_v2(df, win, num_var, task_list, num_targets, num_mag_bins, num_angle_bins):
    """
    fills out a dictionary with key: (task, target, mag_bin, angle_bin)
    each entry is an xarray data array with 3 dimensions: 
    var, time, trial

    updated to check 'task_rot', which has values: 0=co, 1.1=obs+cw, 1.2=obs+ccw

    """
    task_target_bin_dic = {}
    num_win = win[1]-win[0]+1
    for task in task_list: #[0]: 
        for target in range(num_targets): #[0]:
            for bm in range(num_mag_bins):
                for ba in range(num_angle_bins):
                    #identify the number of data points: 
                    task_sel = (df['task'] == task)
                    target_sel = (df['target'] == target)
                    mag_sel = (df['u_v_mag_bin']==bm)
                    angle_sel = (df['u_v_angle_bin']==ba)
                    not_begin = (df['bin']>= -win[0])
                    not_end = (df['bin_end']>= win[1])
                    
                    sel = task_sel&target_sel&mag_sel&angle_sel&not_begin&not_end
                    trial_idxs = np.where(sel)[0]
                    num_trials = len(trial_idxs)
                    
                    #Initialize a nan-filled xarray
                    nan_mat = np.ones((num_var, num_win, num_trials))*np.nan
                    da = xr.DataArray(nan_mat,
                                coords={'var':list(df.columns),
                                                 'time':range(win[0],win[1]+1),
                                                 'trial':range(num_trials)},
                                dims=['var','time','trial'])
                    #Trials: 
                    #-----------------------------------------------------------------------------
                    for i,trial in enumerate(trial_idxs): 
                        trial_data = np.array(df.loc[(trial+win[0]):(trial+win[1]),:]).T
    #                     print(trial_data.shape)
    #                     print(da[:,:,i].shape)
                        da[:,:,i] = trial_data
                    #-----------------------------------------------------------------------------
            
                    #ASSIGN:
                    task_target_bin_dic[task,target,bm,ba] = da
                    task_target_bin_dic[task,target,bm,ba,'num'] = num_trials
                    print(task, target, bm, ba, num_trials)
    return task_target_bin_dic

def calc_command_trials_dic_da(df, win, num_var, num_tasks, num_targets, num_mag_bins, num_angle_bins):
    """
    fills out a dictionary with key: (task, target, mag_bin, angle_bin)
    each entry is an xarray data array with 3 dimensions: 
    var, time, trial

    """
    task_target_bin_dic = {}
    num_win = win[1]-win[0]+1
    for task in range(num_tasks): #[0]: 
        for target in range(num_targets): #[0]:
            for bm in range(num_mag_bins):
                for ba in range(num_angle_bins):
                    #identify the number of data points: 
                    task_sel = (df['task'] == task)
                    target_sel = (df['target'] == target)
                    mag_sel = (df['u_v_mag_bin']==bm)
                    angle_sel = (df['u_v_angle_bin']==ba)
                    not_begin = (df['bin']>-win[0])
                    not_end = (df['bin_end']>win[1])
                    
                    sel = task_sel&target_sel&mag_sel&angle_sel&not_begin&not_end
                    trial_idxs = np.where(sel)[0]
                    num_trials = len(trial_idxs)
                    
                    #Initialize a nan-filled xarray
                    nan_mat = np.ones((num_var, num_win, num_trials))*np.nan
                    da = xr.DataArray(nan_mat,
                                coords={'var':list(df.columns),
                                                 'time':range(win[0],win[1]+1),
                                                 'trial':range(num_trials)},
                                dims=['var','time','trial'])
                    #Trials: 
                    #-----------------------------------------------------------------------------
                    for i,trial in enumerate(trial_idxs): 
                        trial_data = np.array(df.loc[(trial+win[0]):(trial+win[1]),:]).T
    #                     print(trial_data.shape)
    #                     print(da[:,:,i].shape)
                        da[:,:,i] = trial_data
                    #-----------------------------------------------------------------------------
            
                    #ASSIGN:
                    task_target_bin_dic[task,target,bm,ba] = da
                    task_target_bin_dic[task,target,bm,ba,'num'] = num_trials
                    print(task, target, bm, ba, num_trials)
    return task_target_bin_dic

def calc_command_psth(task_target_bin_dic, psth_var, min_trials, num_tasks, num_targets, num_mag_bins, num_angle_bins):
    for task in range(num_tasks): #[0]: 
        for target in range(num_targets): #[0]:
            for bm in range(num_mag_bins):
                for ba in range(num_angle_bins):
                    da = task_target_bin_dic[task,target,bm,ba]
                    num_trials = task_target_bin_dic[task,target,bm,ba,'num']
                    if num_trials >= min_trials:
                        psth = da.loc[psth_var,:,:].mean(axis=2)

                        task_target_bin_dic[task,target,bm,ba,'psth'] = psth

                        #Split into two halves for within-movement comparison: 
                        #get two random halves of trials
                        rnd_order = np.arange(num_trials)
                        np.random.shuffle(rnd_order)
                        half = int(round(num_trials/2))
                        rnd0 = np.sort(rnd_order[:half])
                        rnd1 = np.sort(rnd_order[half:])
                        #
                        psth0 = da.loc[psth_var,:,rnd0].mean(axis=2)
                        psth1 = da.loc[psth_var,:,rnd1].mean(axis=2)
                        #
                        task_target_bin_dic[task,target,bm,ba,'psth_trials',0] = rnd0
                        task_target_bin_dic[task,target,bm,ba,'psth_trials',1] = rnd1
                        task_target_bin_dic[task,target,bm,ba,'psth',0] = psth0
                        task_target_bin_dic[task,target,bm,ba,'psth',1] = psth1

def calc_command_psth_center_at_lag(task_target_bin_dic, psth_var, lag_idx_c, min_trials, num_tasks, num_targets, num_mag_bins, num_angle_bins):
    for task in range(num_tasks): #[0]: 
        for target in range(num_targets): #[0]:
            for bm in range(num_mag_bins):
                for ba in range(num_angle_bins):
                    da = task_target_bin_dic[task,target,bm,ba]
                    num_trials = task_target_bin_dic[task,target,bm,ba,'num']
                    if num_trials >= min_trials:
                        data = da.loc[psth_var,:,:]
                        data_c = center_by_data_at_lag(data, lag_idx_c, lag_axis=1)
                        psth = data_c.mean(axis=2)

                        task_target_bin_dic[task,target,bm,ba,'psth'] = psth

                        #Split into two halves for within-movement comparison: 
                        #get two random halves of trials
                        rnd_order = np.arange(num_trials)
                        np.random.shuffle(rnd_order)
                        half = int(round(num_trials/2))
                        rnd0 = np.sort(rnd_order[:half])
                        rnd1 = np.sort(rnd_order[half:])
                        #
                        data0 = da.loc[psth_var,:,rnd0]
                        data0_c = center_by_data_at_lag(data0, lag_idx_c, lag_axis=1)
                        psth0 = data0.mean(axis=2)

                        data1 = da.loc[psth_var,:,rnd1]
                        data1_c = center_by_data_at_lag(data1, lag_idx_c, lag_axis=1)
                        psth1 = data0.mean(axis=2)

                        #
                        task_target_bin_dic[task,target,bm,ba,'psth_trials',0] = rnd0
                        task_target_bin_dic[task,target,bm,ba,'psth_trials',1] = rnd1
                        task_target_bin_dic[task,target,bm,ba,'psth',0] = psth0
                        task_target_bin_dic[task,target,bm,ba,'psth',1] = psth1                        

def center_by_data_at_lag(data, lag, lag_axis=1):
    """
    Centers a data matrix by data at a specific lag (i.e. index) along a specific axis
    """
    orig_shape = list(data.shape)
    sel_shape=copy.copy(orig_shape)
    sel_shape[lag_axis] = 1
    sel = np.reshape(np.take(np.array(data), lag, lag_axis), sel_shape)
    sel_expand = np.repeat(sel,orig_shape[lag_axis], axis=lag_axis)
    data_c = data-sel_expand
    return data_c


# def calc_command_psth(task_target_bin_dic, psth_var, min_trials, num_tasks, num_targets, num_mag_bins, num_angle_bins):
#     for task in range(num_tasks): #[0]: 
#         for target in range(num_targets): #[0]:
#             for bm in range(num_mag_bins):
#                 for ba in range(num_angle_bins):
#                     da = task_target_bin_dic[task,target,bm,ba]
#                     if task_target_bin_dic[task,target,bm,ba,'num'] >= min_trials:
#                         psth = da.loc[psth_var,:,:].mean(axis=2)
#                         task_target_bin_dic[task,target,bm,ba,'psth'] = psth

def calc_command_psth_diff(task_target_bin_dic, psth_var, task_pairs, zero_lag_idx, min_trials, num_targets, num_mag_bins, num_angle_bins):
    """
    This code calculates the difference between psth's locked to a command. 
    ASSUMES two tasks
    REQUIRES task_target_bin_dic was modified by method 'calc_command_psth' to have the 'psth' entry
    INPUT:
    'task_target_bin_dic' computed from 'calc_command_psth'
    'task_pairs' - list of tuples, each tuple contains the two tasks to compare.  common use would be: [(0,0), (0,1), (1,1)]
    OUTPUT:
    'diff_df' - dataframe 
    """
    columns = ['diff_norm',
                'mag_bin', 'angle_bin', 
                'task0', 'target0', 'num_trials0', 'u_vx0', 'u_vy0',
                'task1', 'target1', 'num_trials1', 'u_vx1', 'u_vy1']
    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    task_pairs = [(0,0), (0,1), (1,1)]
    vec_diff_dic = {}

    for task0, task1 in task_pairs:
        for t0 in range(num_targets):
            if task0 == task1:
                t1_set = range(t0, num_targets)
            else:
                t1_set = range(0,num_targets)
            for t1 in t1_set:
                print(task0, t0, task1, t1)
                for bm in range(num_mag_bins):
                    for ba in range(num_angle_bins):
                        num_trials0 = task_target_bin_dic[task0,t0,bm,ba,'num']
                        d0_valid = num_trials0 >= min_trials
                        num_trials1 = task_target_bin_dic[task1,t1,bm,ba,'num']
                        d1_valid = num_trials1 >= min_trials                        
                        if d0_valid&d1_valid:
                            #Check if same movement: 
                            if (task0==task1)&(t0==t1):
                                #if same movement, compare psth's on different splits of data: 
                                d0_a = task_target_bin_dic[task0,t0,bm,ba,'psth',0]
                                d0 = d0_a.loc[psth_var,:]
                                d1_a = task_target_bin_dic[task0,t0,bm,ba,'psth',1]
                                d1 = d1_a.loc[psth_var,:]
                            else:
                                d0_a = task_target_bin_dic[task0,t0,bm,ba,'psth']
                                d0 = d0_a.loc[psth_var,:]
                                d1_a = task_target_bin_dic[task1,t1,bm,ba,'psth']
                                d1 = d1_a.loc[psth_var,:]

                            #ASSIGN:
                            vec_diff_dic[bm, ba, task0, t0, task1, t1] = d0-d1
                            df_i = copy.copy(nan_df)
                            df_i['diff_norm'] = np.linalg.norm(d0-d1)
                            df_i['mag_bin'] = bm
                            df_i['angle_bin'] = ba

                            df_i['task0'] = task0
                            df_i['target0'] = t0
                            df_i['num_trials0'] = num_trials0
                            df_i['u_vx0'] = float(d0_a.loc['u_vx',zero_lag_idx])
                            df_i['u_vy0'] = float(d0_a.loc['u_vy',zero_lag_idx])
                            
                            df_i['task1'] = task1
                            df_i['target1'] = t1
                            df_i['num_trials1'] = num_trials1
                            df_i['u_vx1'] = float(d1_a.loc['u_vx',zero_lag_idx])
                            df_i['u_vy1'] = float(d1_a.loc['u_vy',zero_lag_idx])
                            #APPEND:
                            df_list.append(df_i)

    diff_df = pd.concat(df_list, ignore_index=True)
    return diff_df, vec_diff_dic

# def calc_command_psth_diff(task_target_bin_dic, min_trials, num_targets, num_mag_bins, num_angle_bins):
#     """
#     This code calculates the difference between psth's locked to a command. 
#     It uses 'task_target_bin_dic'
#     It's hard coded to compare task0 movements to task1 movements
#     output: 
#     da_diff - data array with dimensions: mag_bin, angle_bin, target_task0, target_task1, value: psth difference
#     d_accum - array containing all the differences 
#     """

#     nan_mat = np.ones((num_mag_bins, num_angle_bins, num_targets, num_targets))*np.nan
#     da = xr.DataArray(nan_mat,
#                 coords={'mag':range(num_mag_bins),
#                                 'angle':range(num_angle_bins),
#                                 't0':range(num_targets),
#                                 't1':range(num_targets)},
#                       dims=['mag','angle','t0','t1'])
#     d_accum = []
#     for bm in range(num_mag_bins):
#         for ba in range(num_angle_bins):
#             for t0 in range(num_targets):
#                 task = 0
#                 d0_valid = task_target_bin_dic[task,t0,bm,ba,'num'] >= min_trials
#                 if d0_valid: 
#                     d0 = task_target_bin_dic[task,t0,bm,ba,'psth']            
#                     for t1 in range(num_targets):
#                         task = 1
#                         d1_valid = task_target_bin_dic[task,t1,bm,ba,'num'] >= min_trials
#                         if d1_valid: 
#                             d1 = task_target_bin_dic[task,t1,bm,ba,'psth']
#                             da[bm, ba, t0, t1] = np.linalg.norm(d0-d1)
#                             d_accum.append(da[bm, ba, t0, t1])    
#     da_diff = da
#     d_accum = np.array(d_accum)
#     return da_diff, d_accum

def calc_command_triggered_psth_diff_at_lag(lags, task_target_bin_dic, min_trials, num_mag_bins, num_angle_bins, num_targets, num_tasks):
    """

    """
    #Calc diff of mag and diff of angle: 
    columns = ['diff_mag', 'diff_angle', 'diff_mag_abs', 'diff_angle_abs',
                'diff_norm', 'diff_x', 'diff_y', 'lag',
                'mag_bin_current', 'angle_bin_current', 
                'task0', 'target0', 'num_trials0', 'u_vx0', 'u_vy0',
                'task1', 'target1', 'num_trials1', 'u_vx1', 'u_vy1']
    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    task_pairs = [(0,0), (0,1), (1,1)]
    #Compute:        

    for task0, task1 in task_pairs:
        for t0 in range(num_targets):
            if task0 == task1:
                t1_set = range(t0, num_targets)
            else:
                t1_set = range(0,num_targets)
            # t1_set = range(0,num_targets)
            for t1 in t1_set:
                print(task0, t0, task1, t1)
                for bm in range(num_mag_bins):
                    for ba in range(num_angle_bins):
                        num_trials0 = task_target_bin_dic[task0,t0,bm,ba,'num']
                        d0_valid = num_trials0 >= min_trials
                        num_trials1 = task_target_bin_dic[task1,t1,bm,ba,'num']
                        d1_valid = num_trials1 >= min_trials                        

                        if d0_valid&d1_valid:
                            #Check if same movement: 
                            if (task0==task1)&(t0==t1):
                                #if same movement, compare psth's on different splits of data: 
                                d0 = task_target_bin_dic[task0,t0,bm,ba,'psth',0]
                                d1 = task_target_bin_dic[task0,t0,bm,ba,'psth',1]
                            else:
                                d0 = task_target_bin_dic[task0,t0,bm,ba,'psth']
                                d1 = task_target_bin_dic[task1,t1,bm,ba,'psth']

                            for l in lags:
                                #Convert to polar: 
                                x0 = float(d0.loc['u_vx',l])
                                y0 = float(d0.loc['u_vy',l])
                                d0_mag = np.linalg.norm(np.array(d0.loc[:,l]))                            
                                d0_angle = np.arctan2(y0,x0)
                                
                                x1 = float(d1.loc['u_vx',l])
                                y1 = float(d1.loc['u_vy',l])
                                d1_mag = np.linalg.norm(np.array(d1.loc[:,l]))        
                                d1_angle = np.arctan2(y1,x1)
                                                        
                                l_diff_mag = d0_mag-d1_mag
                                l_diff_angle = center_angle(d1_angle-d0_angle, 0) 
                                l_diff_norm = np.linalg.norm(np.array([x0-x1, y0-y1]))                           
                                
                                #ASSIGN:
                                df_i = copy.copy(nan_df)
                                df_i['diff_mag_abs'] = np.abs(l_diff_mag)
                                df_i['diff_angle_abs'] = np.abs(l_diff_angle)
                                df_i['diff_mag'] = l_diff_mag
                                df_i['diff_angle'] = l_diff_angle
                                df_i['diff_norm'] = l_diff_norm
                                df_i['diff_x'] = float(x0-x1)
                                df_i['diff_y'] = float(y0-y1)
                                df_i['lag'] = l
                                df_i['mag_bin_current'] = bm
                                df_i['angle_bin_current'] = ba
                                
                                df_i['task0'] = task0
                                df_i['target0'] = t0
                                df_i['num_trials0'] = num_trials0
                                df_i['u_vx0'] = x0
                                df_i['u_vy0'] = y0
                                
                                df_i['task1'] = task1
                                df_i['target1'] = t1
                                df_i['num_trials1'] = num_trials1
                                df_i['u_vx1'] = x1
                                df_i['u_vy1'] = y1
                                #APPEND: 
                                df_list.append(df_i)
                            
    diff_df = pd.concat(df_list, ignore_index=True)
    return diff_df  

#
def calc_command_triggered_psth_diff_at_lag_across_task(lags, task_target_bin_dic, min_trials, num_mag_bins, num_angle_bins, num_targets):
    """

    """
    #Calc diff of mag and diff of angle: 
    columns = ['diff_mag', 'diff_angle', 'diff_x', 'diff_y', 'lag',
               'mag_bin_current', 'angle_bin_current', 
               'task0', 'target0', 'num_trials0', 'u_vx0', 'u_vy0',
               'task1', 'target1', 'num_trials1', 'u_vx1', 'u_vy1']
    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    #Compute:        
    for bm in range(num_mag_bins):
        for ba in range(num_angle_bins):
            for t0 in range(num_targets):
                task = 0
                num_trials0 = task_target_bin_dic[task,t0,bm,ba,'num']
                d0_valid = num_trials0 >= min_trials
                if d0_valid: 
                    d0 = task_target_bin_dic[task,t0,bm,ba,'psth']            
                    for t1 in range(num_targets):
                        task = 1
                        num_trials1= task_target_bin_dic[task,t1,bm,ba,'num']
                        d1_valid = num_trials1 >= min_trials
                        if d1_valid: 
                            d1 = task_target_bin_dic[task,t1,bm,ba,'psth']
    #                         psth_diff = d0-d1
                            #loop lags: 
                            for l in lags:
                                #Convert to polar: 
                                x0 = float(d0.loc['u_vx',l])
                                y0 = float(d0.loc['u_vy',l])
                                d0_mag = np.linalg.norm(np.array(d0.loc[:,l]))                            
                                d0_angle = np.arctan2(y0,x0)
                                
                                x1 = float(d1.loc['u_vx',l])
                                y1 = float(d1.loc['u_vy',l])
                                d1_mag = np.linalg.norm(np.array(d1.loc[:,l]))        
                                d1_angle = np.arctan2(y1,x1)
                                                        
                                l_diff_mag = d0_mag-d1_mag
                                l_diff_angle = center_angle(d1_angle-d0_angle, 0)                            
                                
                                #ASSIGN:
                                df_i = copy.copy(nan_df)
                                df_i['diff_mag'] = l_diff_mag
                                df_i['diff_angle'] = l_diff_angle
                                df_i['diff_x'] = float(x0-x1)
                                df_i['diff_y'] = float(y0-y1)
                                df_i['lag'] = l
                                df_i['mag_bin_current'] = bm
                                df_i['angle_bin_current'] = ba
                                
                                df_i['task0'] = 0
                                df_i['target0'] = t0
                                df_i['num_trials0'] = num_trials0
                                df_i['u_vx0'] = x0
                                df_i['u_vy0'] = y0
                                
                                df_i['task1'] = 1
                                df_i['target1'] = t1
                                df_i['num_trials1'] = num_trials1
                                df_i['u_vx1'] = x1
                                df_i['u_vy1'] = y1
                                #APPEND: 
                                df_list.append(df_i)
                                
    diff_df = pd.concat(df_list, ignore_index=True)
    return diff_df  

def plot_hist_stair(bin_edges, data, label=''):
    """
    input: 
    bin_edges: 2 x num_bins
    """
    x = []
    y = []
    for i,b in enumerate(bin_edges.T):
        x.append(b[0])
        x.append(b[1])
        y.append(data[i])
        y.append(data[i])
    if len(label)>0:
        plt.plot(x,y,label=label,linewidth=2)
    else:
        plt.plot(x,y)

def subsample_2datasets_to_match_mean(match_var, d_list, pval_sig, max_iter=5):
    """
    This code subsamples data from two data sets so that their means are matched for all chosen variables
    In future maybe this can be modified to handle more than 2 data sets

    INPUT: 
    match_var: list of variables which should have no significant difference in mean
    d_list: list of data sets.  
    each data set is an xarray
    xarray: num_var X num_observations
    pval_sig: pvalue considered statistically significant

    OUTPUT: 
    DataFrame:
    a data frame with rows for each data_set, columns include: 
    num_kept: number of observations kept after matching procedure
    num_discarded: number of observations discarded during matching procedure

    List:
    kept_list
    discard_list

    XArray:
    ttest_results
    xarray: num_var X num_features
    features: t-stat, pval, mean_init, mean_match

    XArray: 
    mean_results
    xarray: num_var X num_datasets X 

    """
    num_d = len(d_list)
    assert num_d==2, 'Currently mean-matching more than 2 data sets is unsupported!'

    #INITIALIZE:
    #-----------------------------------------------------------------------------------------------------------------------------
    #DataFrame
    columns = ['num_init', 'num_kept', 'num_discarded']
    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    kept_list = []
    discard_list = []
    for i,d in enumerate(d_list):
        df_i = copy.copy(nan_df)
        #ASSIGN:
        num_observations = d.shape[1]
        df_i['num_init'] = num_observations
        df_i['num_kept'] = num_observations
        df_i['num_discarded'] = 0
        df_list.append(df_i)

        kept_list.append(range(num_observations))
        discard_list.append([])
    # for var in match_
    df = pd.concat(df_list, ignore_index=True)

    #XArray ttest
    #NOTE: will need to repeat this at the end of the function:
    num_var = len(match_var)
    num_features = 4
    feature_list = ['tstat_init', 'pval_init', 'tstat_match', 'pval_match']
    nan_mat = np.ones((num_var, num_features))*np.nan
    ttest_r = xr.DataArray(nan_mat, coords={'var':match_var, 'features':feature_list}, dims=['var', 'features'])

    #XArray mean: 
    num_var = len(match_var)
    num_features = 4
    feature_list = ['mean_init', 'var_init', 'mean_match', 'var_match']
    nan_mat = np.ones((num_var, num_d, num_features))*np.nan
    mean_r = xr.DataArray(nan_mat, coords={'var':match_var, 'dataset':np.arange(num_d), 'features':feature_list}, \
        dims=['var', 'dataset', 'features'])

    for var in match_var:
        vd_list = []
        for i,d in enumerate(d_list):
            d_i = np.array(d.loc[var,:])
            vd_list.append(d_i)
            mean_r.loc[var,i,'mean_init'] = d_i.mean()
            mean_r.loc[var,i,'var_init'] = d_i.var()
        (tstat,pval) = sio_stat.ttest_ind(vd_list[0], vd_list[1], equal_var=True)
        ttest_r.loc[var, 'tstat_init'] = tstat
        ttest_r.loc[var, 'pval_init'] = pval
    #-----------------------------------------------------------------------------------------------------------------------------
    #PROCEDURE: 
    #(follows a heuristic, not optimal for sure)

    #Loop over variables
    #Check if there's a significant difference, if not, skip this loop
    #If so, remove data from data set with more data
    
    complete = False
    success = False
    num_iter = 0
    while not complete: 
        mean_equal = []
        for var in match_var:
            vd_list = []
            d_num_obs = []
            for i,d in enumerate(d_list):
                d_i = np.array(d.loc[var, kept_list[i]]) #data only using kept observations
                vd_list.append(d_i)
                d_num_obs.append(len(d_i))
            d_num_obs = np.array(d_num_obs)
            d_big = d_num_obs.argmax()
            d_small = d_num_obs.argmin()
            num_obs_small = d_num_obs.min()
            if num_obs_small > 0:
                (tstat,pval) = sio_stat.ttest_ind(vd_list[0], vd_list[1], equal_var=True)
                sig_diff = (pval <= pval_sig)
                mean_equal.append(not sig_diff)
                if sig_diff:
                    if vd_list[d_big].mean() >= vd_list[d_small].mean():
                        #remove the largest
                        i_discard = kept_list[d_big][vd_list[d_big].argmax()]
                        kept_list[d_big].remove(i_discard)
                        discard_list[d_big].append(i_discard)
                    else:
                        #remove the smallest
                        i_discard = kept_list[d_big][vd_list[d_big].argmin()]
                        kept_list[d_big].remove(i_discard)
                        discard_list[d_big].append(i_discard)
                    df.loc[d_big, 'num_kept'] -= 1
                    df.loc[d_big, 'num_discarded'] += 1

            else: 
                print('Mean Matching Failed :(  A data set lost all its data')
        if all(mean_equal):
            complete = True
            success = True
            print('Mean Matching Succeeded :)')
        elif num_iter == max_iter:
            complete = True
            print('Max Iter Reached')
        num_iter+=1
        # print('num iterations:', num_iter)
        # print('discard_list', discard_list)

    #-----------------------------------------------------------------------------------------------------------------------------
    #Calculate stats after mean matching: 
    for var in match_var:
        vd_list = []
        for i,d in enumerate(d_list):
            sel = kept_list[i]
            d_i = np.array(d.loc[var,sel])
            vd_list.append(d_i)
            mean_r.loc[var,i,'mean_match'] = d_i.mean()
            mean_r.loc[var,i,'var_match'] = d_i.var()
        (tstat,pval) = sio_stat.ttest_ind(vd_list[0], vd_list[1], equal_var=True)
        ttest_r.loc[var, 'tstat_match'] = tstat
        ttest_r.loc[var, 'pval_match'] = pval

    return kept_list, discard_list, df, ttest_r, mean_r


def diff_n_lag0_b_psth_for_command_condition_pair(\
    task_target_bin_dic, task_pairs, num_tasks, num_targets, num_mag_bins, num_angle_bins, min_trials, n_list):
    """
    Code calculates for each (command, condition pair) the difference between neural activity for the command
    and the difference in behavior psth
    """
    columns = ['n_diff_norm', 'b_diff_norm', 'b_diff_norm_lag0',
                'mag_bin', 'angle_bin', 
                'task0', 'target0', 'num_trials0', 'u_vx0', 'u_vy0',
                'task1', 'target1', 'num_trials1', 'u_vx1', 'u_vy1',
                'u_vx_diff_p', 'u_vx_diff_tstat',
                'u_vy_diff_p', 'u_vy_diff_tstat',
                'u_v_mag_diff_p', 'u_v_mag_diff_tstat', 
                'u_v_angle_diff_p', 'u_v_angle_diff_tstat']

    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    # task_pairs = [(0,0), (0,1), (1,1)]
    vec_diff_dic = {}

    for task0, task1 in task_pairs:
        for t0 in range(num_targets):
            if task0 == task1:
                t1_set = range(t0, num_targets)
            else:
                t1_set = range(0,num_targets)
            for t1 in t1_set:
                print(task0, t0, task1, t1)
                for bm in range(num_mag_bins):
                    for ba in range(num_angle_bins):
                        num_trials0 = task_target_bin_dic[task0,t0,bm,ba,'num']
                        d0_valid = num_trials0 >= min_trials
                        num_trials1 = task_target_bin_dic[task1,t1,bm,ba,'num']
                        d1_valid = num_trials1 >= min_trials                        
                        if d0_valid&d1_valid:
                            #Check if same movement: 
                            if (task0==task1)&(t0==t1):
                                #if same movement, compare psth's on different splits of data: 
                                d0 = task_target_bin_dic[task0,t0,bm,ba,'psth',0]
                                d1 = task_target_bin_dic[task0,t0,bm,ba,'psth',1]
                                
                                sel0 = task_target_bin_dic[task0,t0,bm,ba,'psth_trials', 0]
                                dd0 = task_target_bin_dic[task0,t0,bm,ba].loc[:,0,sel0]
                                sel1 = task_target_bin_dic[task0,t0,bm,ba,'psth_trials', 1]
                                dd1 = task_target_bin_dic[task1,t1,bm,ba].loc[:,0,sel1]
                            else:
                                d0 = task_target_bin_dic[task0,t0,bm,ba,'psth']
                                d1 = task_target_bin_dic[task1,t1,bm,ba,'psth']
                                                        
                                dd0 = task_target_bin_dic[task0,t0,bm,ba].loc[:,0,:]
                                dd1 = task_target_bin_dic[task1,t1,bm,ba].loc[:,0,:]                          
                                
                            #ASSIGN:
                            vec_diff_dic[bm, ba, task0, t0, task1, t1] = d0-d1
                            df_i = copy.copy(nan_df)                        
                            
                            #neural diff is over lag 0: 
                            df_i['n_diff_norm'] = np.linalg.norm(d0.loc[n_list,0]-d1.loc[n_list,0])
                            
                            #behavior diff is over all lags: 
                            df_i['b_diff_norm'] = np.linalg.norm(d0.loc[['u_vx', 'u_vy'],:]-d1.loc[['u_vx', 'u_vy'],:])
                            
                            #behavior diff is over all lags: 
                            df_i['b_diff_norm_lag0'] = np.linalg.norm(d0.loc[['u_vx', 'u_vy'],0]-d1.loc[['u_vx', 'u_vy'],0])                        
                            
                            df_i['mag_bin'] = bm
                            df_i['angle_bin'] = ba                 
                            
                            df_i['task0'] = task0
                            df_i['target0'] = t0
                            df_i['num_trials0'] = num_trials0
                            df_i['u_vx0'] = float(d0.loc['u_vx',0])
                            df_i['u_vy0'] = float(d0.loc['u_vy',0])

                            df_i['task1'] = task1
                            df_i['target1'] = t1
                            df_i['num_trials1'] = num_trials1
                            df_i['u_vx1'] = float(d1.loc['u_vx',0])
                            df_i['u_vy1'] = float(d1.loc['u_vy',0])
                            
                            #Check if behavior is significantly different within bin: 
                            #XY:
                            #X:
                            x0 = np.array(dd0.loc['u_vx', :])
                            x1 = np.array(dd1.loc['u_vx', :])
    #                         print(x0.shape, x1.shape)
                            (tstat,pval) = scipy.stats.ttest_ind(x0, x1, equal_var=True)
                            df_i['u_vx_diff_p'] = pval
                            df_i['u_vx_diff_tstat'] = tstat
                            #Y:
                            y0 = np.array(dd0.loc['u_vy', :])
                            y1 = np.array(dd1.loc['u_vy', :])
                            (tstat,pval) = scipy.stats.ttest_ind(y0, y1, equal_var=True)
                            df_i['u_vy_diff_p'] = pval
                            df_i['u_vy_diff_tstat'] = tstat                          
                            
                            #mag,angle:
                            #X:
                            x0 = np.array(dd0.loc['u_v_mag', :])
                            x1 = np.array(dd1.loc['u_v_mag', :])
    #                         print(x0.shape, x1.shape)
                            (tstat,pval) = scipy.stats.ttest_ind(x0, x1, equal_var=True)
                            df_i['u_v_mag_diff_p'] = pval
                            df_i['u_v_mag_diff_tstat'] = tstat
                            #Y:
                            y0 = np.array(dd0.loc['u_v_angle_ctr_bin', :])
                            y1 = np.array(dd1.loc['u_v_angle_ctr_bin', :])
                            (tstat,pval) = scipy.stats.ttest_ind(y0, y1, equal_var=True)
                            df_i['u_v_angle_diff_p'] = pval
                            df_i['u_v_angle_diff_tstat'] = tstat                         
                            
                            #APPEND:
                            df_list.append(df_i)

    diff_df = pd.concat(df_list, ignore_index=True)
    return diff_df    

def diff_df_sel(diff_df, min_trials_list, num_mag_bins, p_sig=0.05):
    """
    takes the output of 'diff_n_lag0_b_psth_for_command_condition_pair'
    and calculates selection filters on the dataframe for use in analysis

    Assumes task 0 = 'co' (center-out)
    Assumes task 1 = 'obs' (obstacle)

    """

    sel_dic = {}

    #TRIALS:
    for min_trials in min_trials_list:
        num_trials0_sel = (diff_df['num_trials0']>=min_trials) #5
        num_trials1_sel = (diff_df['num_trials1']>=min_trials) #5
        sel_dic['num_trials',min_trials] = \
            num_trials0_sel\
            &num_trials1_sel\

    
    #MOVEMENT:
    move_list = ['within_move', 'within_task', 'within_co', 'within_obs', 'across_task', 'across_move']
    sel_dic['within_move'] = (diff_df['target0']==diff_df['target1'])&(diff_df['task0']==diff_df['task1'])
    sel_dic['within_task'] = (diff_df['target0']!=diff_df['target1'])&(diff_df['task0']==diff_df['task1'])
    sel_dic['within_co'] = (diff_df['target0']!=diff_df['target1'])&(diff_df['task0']==diff_df['task1'])&(diff_df['task0']==0)
    sel_dic['within_obs'] = (diff_df['target0']!=diff_df['target1'])&(diff_df['task0']==diff_df['task1'])&(diff_df['task0']==1)
    sel_dic['across_task'] = (diff_df['task0']!=diff_df['task1'])
    sel_dic['across_move'] = (diff_df['task0']!=diff_df['task1'])|(diff_df['target0']!=diff_df['target1'])

    #MAGNITUDE
    mag_list = range(num_mag_bins) #[0,1,2,3]
    for bm in mag_list:
        sel_dic['mag', bm] = (diff_df['mag_bin']==bm)
        
    #SIGNIFANCE
    # p_sig = 0.05
    sel_dic['x_y_sig'] = \
    (diff_df['u_vx_diff_p'] <= p_sig)\
    |(diff_df['u_vy_diff_p'] <= p_sig)

    sel_dic['mag_angle_sig'] = \
    (diff_df['u_v_mag_diff_p'] <= p_sig)\
    |(diff_df['u_v_angle_diff_p'] <= p_sig)

    return sel_dic, move_list

def preprocess_bmi_df(df, target_pos, num_prefix, num_tasks, num_targets):
    """
    adds the following "preprocessing" columns to the data frame: 

    'bin_end': number of samples till you reach the last sample of the trial
    'prog': progress till end of trial (ranges from 0 to 1)
    'trial_cond': 
    POLAR:
    p_mag, p_angle, v_mag, v_angle, u_p_mag, u_p_angle, u_v_mag, u_v_angle
    CW vs CCW
    
    """
    #Pre-processing: 

    #Trial boundaries:
    trial_start = np.where((df['bin']==-10.0))[0]
    trial_stop = np.where((df['trial_stop']==1))[0]
    trial_bound = np.vstack((trial_start,trial_stop)).T
    num_trials = trial_bound.shape[0]

    #-----------------------------------------------------------------------------------------------
    #Time till end of trial: 
    df['bin_end'] = 0
    df['prog'] = 0

    for bnd in trial_bound:
        bin_data = df.loc[bnd[0]:bnd[1], 'bin']
        last_bin = bin_data.iloc[-1]
        bin_end = last_bin-bin_data
        prog = bin_data/last_bin
        #ASSIGN:
        df.loc[bnd[0]:bnd[1], 'bin_end'] = bin_end
        df.loc[bnd[0]:bnd[1], 'prog'] = prog
    
    #-----------------------------------------------------------------------------------------------    
    #Cond Trial number
    for task in range(num_tasks):
        for target in range(num_targets):
            cond_sel = (df['task']==task) & (df['target']==target)
            trial_start = (df['bin']==0) & cond_sel 
            trial_stop = (df['bin_end']==0) & cond_sel
            trial_bnd = np.vstack((np.where(trial_start)[0], np.where(trial_stop)[0]))
            for i,bnd in enumerate(trial_bnd.T):
                # print(i, bnd)
                df.loc[bnd[0]:bnd[1], 'trial_cond'] = i
    
    #-----------------------------------------------------------------------------------------------    
    #Polar coordinates: 
    # 1) Convert stuff to polar, 2) calculate distance to target
    # 1) Convert stuff to polar
    df['p_mag'], df['p_angle'] = cartesian2polar(df['kin_py'], df['kin_px'])
    df['v_mag'], df['v_angle'] = cartesian2polar(df['kin_vy'], df['kin_vx'])
    df['u_p_mag'], df['u_p_angle'] = cartesian2polar(df['u_py'], df['u_px'])
    df['u_v_mag'], df['u_v_angle'] = cartesian2polar(df['u_vy'], df['u_vx'])

    # 2) distance to target
    error = df.loc[:, 'kin_px':'kin_py']-target_pos[df['target'].astype(int),:]
    df['d2target'] = np.linalg.norm(error,ord=2,axis=1)
    df['x_error'] = error.loc[:,'kin_px']
    df['y_error'] = error.loc[:,'kin_py']

    #-----------------------------------------------------------------------------------------------    
    #CW vs CCW:
    df_determine_cw_ccw(df, target_pos)     


def df_determine_cw_ccw(df, target_pos):
    #Identify if each trajectory is more clockwise or counterclockwise around the axis from center to target: 

    df['task_rot'] = df['task']#0:co, 1.1: obs, cw, 1.2: obs,ccw
    df['cw'] = np.zeros((df.shape[0])) #cw = positive signed area
    df['target_axis_signed_area'] = np.zeros((df.shape[0]))
    df['target_axis_area'] = np.zeros((df.shape[0]))

    trial_start = np.where((df['trial_start']==1))[0]
    trial_stop = np.where((df['trial_stop']==1))[0]
    trial_bound = np.vstack((trial_start,trial_stop)).T
    num_trials = trial_bound.shape[0]

    for bnd in trial_bound: 
        x = df['kin_px'][bnd[0]:bnd[1]+1]
        y = df['kin_py'][bnd[0]:bnd[1]+1]
        t = int(df['target'][bnd[0]])
        t_pos = target_pos[t,:]
        sa, a, df_a = traj_signed_area_about_target_axis(x, y, t_pos)
        if sa>0:
            cw=1
        else:
            cw=0
        # #debug: 
        # tsk = task_list[int(df['task'][bnd[0]])]
        # trl = df['trial'][bnd[0]]
        # trl_len = bnd[1]+1-bnd[0]
        # print(bnd[0], bnd[1], trl_len, tsk, t, trl, sa, cw, a)
        #Insert the data: 
        df['target_axis_signed_area'][bnd[0]:bnd[1]+1] = sa
        df['cw'][bnd[0]:bnd[1]+1] = cw
        df['target_axis_area'][bnd[0]:bnd[1]+1] = a
    
    sel_obs_cw = (df['task']==1)&(df['cw']==0)
    df.loc[sel_obs_cw, 'task_rot'] = 1.1
    sel_obs_cw = (df['task']==1)&(df['cw']==1)
    df.loc[sel_obs_cw, 'task_rot'] = 1.2




def def_command_bin(df, mag_bin_perc=np.array([0,25,50,75,100]), num_angle_bins=8, T0_angle=-3*(2*np.pi)/8):
    """
    FUNCTION:
    defines bins for commands ('u_vx', 'u_vy')
    INPUT:
    mag_bin_perc: numpy array of length num_mag_bins+1
    boundaries of command magnitude (percentiles over all data)
    num_angle_bins: number of angle bins
    T0_angle: the angle corresponding to target 0.  function places the 0th angle bin at this angle.
    """

    #1) magnitude bins: 
    #USUAL:
    mag_data = df['u_v_mag']
    #mag_data = df['u_v_mag'][df['bin']>=0] - we didn't do this, because we want to be able to bin all data, negative bins
    mag_bin = np.percentile(mag_data, mag_bin_perc)
    mag_bin_edges = np.vstack((mag_bin[0:-1], mag_bin[1:]))
    mag_bin_c = mag_bin_edges.mean(axis=0)

    #2) angle bins: 
    angle_bin_c = np.linspace(T0_angle, T0_angle+np.pi*2, num=num_angle_bins+1, endpoint=True)
    angle_bin = angle_bin_c-np.pi*2/16.0
    angle_bin_edges = np.vstack((angle_bin[0:-1], angle_bin[1:]))

    return mag_bin, mag_bin_edges, mag_bin_c, angle_bin_c, angle_bin, angle_bin_edges

def df_center_angle_for_binning(df, angle_bin):
    """
    FUNCTION:
    centers angle variables based on the start and end angle of angle bins

    INPUT:
    OUTPUT:
    """
    #center angles for binning: 
    angle_center_for_binning = (angle_bin[-1]+angle_bin[0])/2.0
    print('angle_center:', angle_center_for_binning*180/np.pi)

    angle_vars = ['p_angle', 'v_angle', 'u_p_angle', 'u_v_angle']
    for d in angle_vars:
        df[d] = center_angle(np.array(df[d]), angle_center_for_binning)
    print('min centered angle:', np.min(df['u_v_angle'])*180/np.pi)
    print('max centered angle:', np.max(df['u_v_angle'])*180/np.pi)
    return angle_center_for_binning

def df_bin_command(df, mag_bin_edges, angle_bin_edges):
    """
    FUNCTION:
    bins the commands ['u_v_mag', 'u_v_angle']

    """
    bin_dic = {}
    bin_dic[0] = mag_bin_edges
    bin_dic[1] = angle_bin_edges

    data2bin = np.array(df[['u_v_mag','u_v_angle']])
    bin_r, hist_r = bin_vec_data(data2bin, bin_dic)
    df['u_v_mag_bin']=bin_r[:,0]
    df['u_v_angle_bin']=bin_r[:,1]

def center_df_angle(df, angle_bin_c, target_angle):

        # 2) Center @ target angle: 
    d_list = ['p_angle', 'v_angle', 'u_p_angle', 'u_v_angle']
    for d in d_list:
        data = df[d]
        t_angle = target_angle[df['target'].astype(int)]
        df[d+'_ctr_t'] = center_angle(df[d], t_angle)

    #-----------------------------------------------------------------------------------------------
    #Center Angle to Bin Angle: 
    d_list = ['u_v_angle']
    for d in d_list:
        data = df[d]
        bin_angle = angle_bin_c[df['u_v_angle_bin'].astype(int)]
        df[d+'_ctr_bin'] = center_angle(df[d], bin_angle)    

def shuffle_df_by_command(df, var_shuffle, var_shuffle_src, num_mag_bins, num_angle_bins, angle_bin_c, target_angle):
    """
    shuffles 'var_shuffle'
    saves where the shuffled data came from 'var_shuffle_src'
    """
    df_S = copy.deepcopy(df)

    var_src_assign = [i+'_shuffle' for i in var_shuffle_src]
    df2concat = pd.DataFrame(columns=var_src_assign)
    df_S = pd.concat([df_S, df2concat], sort=False)

    for bm in range(num_mag_bins):
        for ba in range(num_angle_bins):
            sel = \
            (df_S.loc[:,'u_v_mag_bin']==bm)\
            &(df_S.loc[:,'u_v_angle_bin']==ba)\
            &(df_S.loc[:,'bin']>=0)

            var = var_shuffle+var_shuffle_src
            temp = df_S.loc[sel, var].sample(frac=1).reset_index(drop=True)

            df_S.loc[sel, var_shuffle] = np.array(temp.loc[:,var_shuffle])

            test_assign = np.array(temp.loc[:,var_shuffle_src])
            print(test_assign.shape)
            df_S.loc[sel, var_src_assign] = test_assign

    #Need to recalculate after shuffle: 
    center_df_angle(df_S, angle_bin_c, target_angle)
    return df_S

def df_idx_win2psth_mat(df, idx, win, psth_var):
    """
    code creates a psth matrix (var X time X trials) using a dataframe, idx of events, and window around event idx
    Input:
    df containing data from trials
    idx - idxs of events to lock to
    win - 2 entry vector defining how many samples before and after idxs to include
    psth_var - variables to calculate the psth on
    Output:
    da - xarray data array of (var X time X trials), where var is all var in the df
    psth, psth_sem - xarray of (var X time), where var is just the 'psth_var'

    """
    time = range(win[0],win[1]+1)
    num_win = win[1]-win[0]+1
    num_var = len(list(df.columns))

    #Initialize a nan-filled xarray
    num_trials = len(idx)
    nan_mat = np.ones((num_var, num_win, num_trials))*np.nan
    da = xr.DataArray(nan_mat,
                coords={'var':list(df.columns),
                                 'time':time,
                                 'trial':idx},
                dims=['var','time','trial'])
    #Trials: 
    #-----------------------------------------------------------------------------
    for i,trial in enumerate(idx): 
        trial_data = np.array(df.loc[(trial+win[0]):(trial+win[1]),:]).T
        da[:,:,i] = trial_data
    #-----------------------------------------------------------------------------        
    
    #RESULTS:
    psth,psth_var,psth_sem = bmi_util.da_mean_var_sem(da.loc[psth_var,:,:], axis=2)

    # psth = da.loc[psth_var,:,:].mean(axis=2)
    # psth_sem = sio_stat.sem(da.loc[psth_var,:,:], axis=2)
    # psth_sem = xr.DataArray(psth_sem, 
    #                             coords={'var':psth_var,
    #                                   'time':time},
    #                             dims=['var','time'])
    return da, psth, psth_sem

def subsample_2datasets_to_match_mean_v2(match_var, d_list, match_perc, max_iter=5):
    """
    This code subsamples data from two data sets so that their means are matched for all chosen variables
    (Code subsamples from the larger dataset.)
    In future maybe this can be modified to handle more than 2 data sets

    Subsampling Approach: 
    form a 'cost' for each sample, and remove the sample with highest cost.
    Cost: There is a cost associated with each dimension of data.  
    Cost for each dimension: abs(match_mean-data)
    We sum the cost over all dimensions which are not matched.
    Removal: Remove the sample with highest cost.

    INPUT: 
    match_var: list of variables which should have no significant difference in mean
    d_list: list of data sets.  
    each data set is an xarray
    xarray: num_var X num_observations
    match_perc: percentage difference in mean tolerated 

    OUTPUT: 
    DataFrame:
    a data frame with rows for each data_set, columns include: 
    num_kept: number of observations kept after matching procedure
    num_discarded: number of observations discarded during matching procedure

    List:
    kept_list
    discard_list

    XArray:
    ttest_results
    xarray: num_var X num_features
    features: t-stat, pval, mean_init, mean_match

    XArray: 
    mean_results
    xarray: num_var X num_datasets X 

    """

    num_d = len(d_list)
    assert num_d==2, 'Currently mean-matching more than 2 data sets is unsupported!'

    #INITIALIZE:
    #-----------------------------------------------------------------------------------------------------------------------------
    #DataFrame
    #Each row is a data set, each column is a property of the subsampling
    columns = ['num_init', 'num_kept', 'num_discarded']
    num_col = len(columns)
    nan_df = pd.DataFrame(np.ones((1,num_col))*np.nan, columns=columns)
    df_list = []
    kept_list = []
    discard_list = []
    for i,d in enumerate(d_list):
        df_i = copy.copy(nan_df)
        #ASSIGN:
        num_observations = d.shape[1]
        df_i['num_init'] = num_observations
        df_i['num_kept'] = num_observations
        df_i['num_discarded'] = 0
        df_list.append(df_i)

        kept_list.append(range(num_observations))
        discard_list.append([])
    df = pd.concat(df_list, ignore_index=True)

    #XArray ttest
    #NOTE: will need to repeat this at the end of the function:
    num_var = len(match_var)
    num_features = 4
    feature_list = ['tstat_init', 'pval_init', 'tstat_match', 'pval_match']
    nan_mat = np.ones((num_var, num_features))*np.nan
    ttest_r = xr.DataArray(nan_mat, coords={'var':match_var, 'features':feature_list}, dims=['var', 'features'])

    #XArray mean: 
    num_var = len(match_var)
    num_features = 4
    feature_list = ['mean_init', 'var_init', 'mean_match', 'var_match']
    nan_mat = np.ones((num_var, num_d, num_features))*np.nan
    mean_r = xr.DataArray(nan_mat, coords={'var':match_var, 'dataset':np.arange(num_d), 'features':feature_list}, \
        dims=['var', 'dataset', 'features'])

    for var in match_var:
        vd_list = []
        for i,d in enumerate(d_list):
            d_i = np.array(d.loc[var,:])
            vd_list.append(d_i)
            mean_r.loc[var,i,'mean_init'] = d_i.mean()
            mean_r.loc[var,i,'var_init'] = d_i.var()
        (tstat,pval) = sio_stat.ttest_ind(vd_list[0], vd_list[1], equal_var=True)
        ttest_r.loc[var, 'tstat_init'] = tstat
        ttest_r.loc[var, 'pval_init'] = pval

    #-----------------------------------------------------------------------------------------------------------------------------
    #PROCEDURE: 
    
    complete = False
    success = False
    num_iter = 0
    while not complete: 
        mean_equal = []
        for var in match_var:
            #HERE!!!!  UPDATE CODE



            vd_list = []
            d_num_obs = []
            for i,d in enumerate(d_list):
                d_i = np.array(d.loc[var, kept_list[i]]) #data only using kept observations
                vd_list.append(d_i)
                d_num_obs.append(len(d_i))
            d_num_obs = np.array(d_num_obs)
            d_big = d_num_obs.argmax()
            d_small = d_num_obs.argmin()
            num_obs_small = d_num_obs.min()
            if num_obs_small > 0:
                (tstat,pval) = sio_stat.ttest_ind(vd_list[0], vd_list[1], equal_var=True)
                sig_diff = (pval <= pval_sig)
                mean_equal.append(not sig_diff)
                if sig_diff:
                    if vd_list[d_big].mean() >= vd_list[d_small].mean():
                        #remove the largest
                        i_discard = kept_list[d_big][vd_list[d_big].argmax()]
                        kept_list[d_big].remove(i_discard)
                        discard_list[d_big].append(i_discard)
                    else:
                        #remove the smallest
                        i_discard = kept_list[d_big][vd_list[d_big].argmin()]
                        kept_list[d_big].remove(i_discard)
                        discard_list[d_big].append(i_discard)
                    df.loc[d_big, 'num_kept'] -= 1
                    df.loc[d_big, 'num_discarded'] += 1

            else: 
                print('Mean Matching Failed :(  A data set lost all its data')
        if all(mean_equal):
            complete = True
            success = True
            print('Mean Matching Succeeded :)')
        elif num_iter == max_iter:
            complete = True
            print('Max Iter Reached')
        num_iter+=1
        # print('num iterations:', num_iter)
        # print('discard_list', discard_list)    