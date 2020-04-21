import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

cmap_list = ['orangered', 'olivedrab', 'teal', 'midnightblue', 'darkmagenta']

def plot_strobed(dat, event=5):
    strobed = dat['Strobed']
    ix = np.nonzero(strobed[:, 1]==event)[0]

    pos = np.hstack((dat['AD39'], dat['AD40']))
    plt.plot(pos[:, 0], pos[:, 1])

    strobed_ts = (strobed[ix, 0]*1000).astype(int)
    plt.plot(pos[strobed_ts, 0], pos[strobed_ts, 1], 'r.')

def check_strobed_obs(dat):
    # plot time diff between 3 obstacle appearances
    strobed = dat['Strobed']

    # Trial list: 
    trialList = np.zeros((11, 3))
    trialList[1,:]  = [5, 2, 11]
    trialList[2,:]  = [11, 2, 5]
    trialList[3,:]  = [3, 13, 11]
    trialList[4,:]  = [11, 13, 3]
    trialList[5,:]  = [5, 16, 9]
    trialList[6,:]  = [9, 16, 5]
    trialList[7,:]  = [3, 21, 9]
    trialList[8,:]  = [9, 21, 3]
    trialList[9,:]  = [23, 20, 24]
    trialList[10,:] = [24, 20, 23]

    trialList = trialList + 63
    trialList = trialList[1:, :]

    # Find trial indices: 
    ix = np.nonzero(strobed[:, 1] == 15)[0]

    # Get position: 
    pos = np.hstack((dat['AD39'], dat['AD40']))

    # Low pass filter this: 
    N, Wn = scipy.signal.buttord(100/500., 150/500., 3, 20)
    b, a, = scipy.signal.butter(N, 100/500., 'lowpass')
    pos_filt = scipy.signal.filtfilt(b, a, pos, axis=0)

    #Characterize each trial
    trial_ix = np.zeros((len(ix), ))
    ts_offs = np.zeros((len(ix), 2))
    for i in range(len(ix)):
        tgs = np.tile(strobed[ix[i]-3:ix[i], 1][np.newaxis, :], [10, 1])
        ts_offs[i, :] = np.diff(strobed[ix[i]-1:ix[i]+2, 0])

        trl = np.nonzero(np.sum(np.abs(trialList - tgs), axis=1) == 0)[0]
        trial_ix[i] = trl

    # Plot each trial in a different color
    f, ax = plt.subplots(ncols=5)

    for i, I in enumerate(strobed[ix, 0]):
        tix = range(int(I*1000), int((I+3)*1000))
        trl = int(trial_ix[i])/2
        ax[trl].plot(pos_filt[tix, 0], pos_filt[tix, 1], '-', color=cmap_list[trl])
    for i in range(5):
        ax[i].set_xlim([-.4, 1.4])
        ax[i].set_ylim([.5, 2.2])
    return ts_offs






