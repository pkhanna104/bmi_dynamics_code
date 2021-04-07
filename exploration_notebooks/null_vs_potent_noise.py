

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt


def run_all(nneurons=44, nT=10000, noise_scale=1):

    KG = np.random.randn(2, nneurons)
    KG = KG / np.linalg.norm(KG, axis=1)[:, np.newaxis]

    f, ax = plt.subplots(nrows = 3, figsize = (6, 6))

    ax[0].set_ylabel('Full Diffs')
    ax[1].set_ylabel('Null Diffs')
    ax[2].set_ylabel('Potent Diffs')

    for noisetype in ['none', 'null', 'potent', 'full']:
        run(nT, KG, ax, noisetype=noisetype, noise_scale=noise_scale)

    ax[0].set_title('nT = %d, noise_scale=%1.f, nneurons = %d'%(nT, noise_scale,
        nneurons))
    plt.legend()


def run(nT, KG, ax, noisetype = None, noise_scale = 5): 
    
    NNeur = KG.shape[1]
    Ndim = KG.shape[0]
    min_obs = 15

    ### Null -- generate a set of potent commands ###
    activity = np.random.randn(NNeur, nT)
    commands = np.dot(KG, activity)

    ### Remove null component; 
    KG_null = scipy.linalg.null_space(KG) # N x (N-2)
    KG_null_proj = np.dot(KG_null, KG_null.T)
    activity_null = np.dot(KG_null_proj, activity)

    assert(np.allclose(np.dot(KG, activity_null), 0.))
    assert(np.allclose(np.dot(KG, activity - activity_null), commands))

    #### If noisetype 
    noise = np.random.randn(NNeur, nT)*noise_scale
    if noisetype == 'null':
        noise_null = np.dot(KG_null_proj, noise)
        activity = activity + noise_null
        color='g'

    elif noisetype == 'potent':
        noise_null = np.dot(KG_null_proj, noise)
        activity = activity + (noise - noise_null)
        color = 'r'

    elif noisetype == 'full':
        activity = activity + noise 
        color = 'b'

    elif noisetype == 'none':
        color='k'

    ##### Re0make commands after adding noise ###
    commands = np.dot(KG, activity)

    ### 
    activity_null = np.dot(KG_null_proj, activity)
    activity_pot = activity - activity_null
    assert(np.allclose(np.dot(KG, activity_pot), commands))
    assert(np.allclose(np.dot(KG, activity), commands))
    assert(np.allclose(np.dot(KG, activity_null), 0.))

    ### Bin commands 
    command_mag = np.linalg.norm(commands, axis=0)
    command_ang = np.array([math.atan2(y, x) for _, (x, y) in enumerate(commands.T)])

    ### Randomly assign movements 
    movements = np.random.randint(0, 10, (nT))

    ### Do command movement-analysis 
    bin_mag, bin_ang = bin_commands(command_ang, command_mag)
    diff_dist = []
    diff_null = []
    diff_pot = []

    for mag in range(4):
        for ang in range(8):

            ix_com = np.nonzero(np.logical_and(bin_mag == mag, bin_ang == ang))[0]

            if len(ix_com) > 0:
                assert(np.allclose(np.dot(KG, activity_pot[:, ix_com]), commands[:, ix_com]))
                assert(np.allclose(np.dot(KG, activity[:, ix_com]), commands[:, ix_com]))
                assert(np.allclose(np.dot(KG, activity[:, ix_com]), np.dot(KG, activity_pot[:, ix_com])))
                assert(np.allclose(np.mean(np.dot(KG, activity[:, ix_com]), axis=1), 
                    np.mean(np.dot(KG, activity_pot[:, ix_com]), axis=1)))

                ### global mean
                global_mn =     np.mean(activity     [:, ix_com], axis=1)
                global_null_mn =np.mean(activity_null[:, ix_com], axis=1)
                global_pot_mn = np.mean(activity_pot [:, ix_com], axis=1)
                
                assert(np.allclose(np.dot(KG, global_mn), np.dot(KG, global_pot_mn)))
                assert(np.allclose(np.dot(KG, global_null_mn), 0.))

                for mov in np.unique(movements[ix_com]):

                    ix_mov= np.nonzero(movements[ix_com] == mov)[0]

                    if len(ix_mov) > min_obs:

                        mov_com = np.mean(activity[:, ix_com[ix_mov]], axis=1)
                        mov_com_null = np.mean(activity_null[:, ix_com[ix_mov]], axis=1)
                        mov_com_pot =  np.mean(activity_pot[:, ix_com[ix_mov]], axis=1)
                        
                        diff_dist.append(np.linalg.norm(mov_com - global_mn))
                        diff_null.append(np.linalg.norm(mov_com_null - global_null_mn))
                        diff_pot.append(np.linalg.norm(mov_com_pot - global_pot_mn))

    #### Plot distributions ####
    for i_dist, (axi, dist) in enumerate(zip(ax, [diff_dist, diff_null, diff_pot])):
        h, i = np.histogram(dist)
        axi.plot(i[:-1] + 0.5*(i[1]-i[0]), h/float(np.sum(h)),color+'-', label=noisetype)
        axi.vlines(np.mean(dist), 0, .5, color)
    
    return ax


def bin_commands(command_ang, command_mag):

    mag_bins = np.array([0., np.percentile(command_mag, 25), np.percentile(command_mag, 50),
        np.percentile(command_mag, 75), np.percentile(command_mag, 100)+1])

    ang_bins = np.linspace(-np.pi/8, 2*np.pi + np.pi/8, 10)

    command_ang[command_ang < 0] += 2*np.pi
    command_ang[command_ang < ang_bins[1]] += 2*np.pi

    ix_bin = np.digitize(command_ang, ang_bins)
    ix_bin[ix_bin == 9] == 1
    ix_bin = ix_bin - 1

    ix_bin_mag = np.digitize(command_mag, mag_bins)
    ix_bin_mag = ix_bin_mag - 1

    return ix_bin_mag, ix_bin