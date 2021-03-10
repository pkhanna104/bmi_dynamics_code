import resim
import tables
import pickle 

# hdf_file = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210103_06_te3210.hdf'
# decoder = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210103_00_home20210103_01_te3205_KFDecoder.pkl'

# hdf = tables.openFile(hdf_file)
# dec = pickle.load(open(decoder))

# R = resim.RerunDecoding(hdf, dec, task='bmi_multi')

# sc = hdf.root.task[:]['spike_counts']
# R.run_decoder(sc, False)
# resim.plot_traj(R)

# hdf_file = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210303_31_te3369.hdf'
# decoder =  '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210303_28_test03031547.pkl'

hdf_file = '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210304_06_te3380.hdf'
decoder =  '/Users/preeyakhanna/Dropbox/TimeMachineBackups/home2020/home20210304_03_home20210304_04_te3378_KFDecoder.pkl'

hdf = tables.openFile(hdf_file)
dec = pickle.load(open(decoder))

R = resim.RerunDecoding(hdf, dec, task='bmi_multi')

sc = hdf.root.task[:]['spike_counts']
R.run_decoder(sc, False)
resim.plot_traj(R)

f, ax = plt.subplots()
ax.plot(R.cursor_vel[:, 0], 'k-')
ax.plot(R.dec_state_mn['true'][:, 3], 'b-')
ax.plot(R.dec_state_mn['all'][:, 3], 'r-')

