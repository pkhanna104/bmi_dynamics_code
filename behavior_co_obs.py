import scipy.io as sio
import numpy as np
import pandas as pd
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
def center_angle(angle, ctr_orig, ctr_new):
    """
    ctr_orig: the center angle of angle data.  
    e.g. if angle ranges from (0,2*pi) then the center is pi.  
    if angle ranges from (-pi,pi) then the center is 0.
    """
    angle_center = (angle+(np.pi-ctr_orig)-ctr_new)%(2*np.pi)-(np.pi-ctr_orig)
    return angle_center   

#COMMENT: This code is the one to use... not center_angle...
def center_angle_v2(angle, ctr):
    """
    angle: array-like
    ctr: the center angle of angle data.  
    e.g. if you want angle to range from (0,2*pi) then the center is pi.  
    if you want angle to range from (-pi,pi) then the center is 0.
    """
    angle_center = angle+np.matrix.round((ctr-angle)/(2*np.pi))*2*np.pi
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