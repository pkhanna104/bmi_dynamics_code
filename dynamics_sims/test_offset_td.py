import numpy as np
import matplotlib.pyplot as plt 
import plot_flow_field_utils as ffu
pref_colors = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 
		'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black', 'gray']

def test_methods_of_ffplts():
	'''
	Method to test tanners ways of plotting b
	Validating accuracy by comparing methods and plotting fixed points
	Also plotting noiseless data on each plot 
	''' 

	### Rotationmatirx 
	theta = np.pi/12.
	A = np.array([[np.cos(theta), -1*np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	b = np.array([[1.], [4.]])

	### flwo field plot lims
	lims = 20

	### Simulate Data nad plot it in the top / left  
	Xt = np.random.randn(10, 2)
	f, ax = plt.subplots(ncols = 3, nrows = 2, figsize=(10, 6))
	for ic, c in enumerate(pref_colors): 
		ax[0, 0].plot(Xt[ic, 0], Xt[ic, 1], '.', markersize=20, color=c)
		xt = Xt[ic, :]
		X_tmp = [xt]

		for t in range(10): 
			xt = np.dot(A, xt[:, np.newaxis]) + b
			xt = np.squeeze(xt)
			X_tmp.append(xt)

		X_tmp = np.vstack((X_tmp))
		ax[0, 0].plot(X_tmp[:, 0], X_tmp[:, 1], '.-', color=c, alpha=0.5)

	ax[0, 0].set_title('Data: X_{t+1} = A X_t + b')

	######################## Method 1 ##########################################
	######################## Plot flow field from A+B appended #################
	#### A_app = [[A b]; [zeros, 1]], state is X_app_t = [X_t; 1]
	########## Plot in X-space ##########
	A_app = np.hstack((A, b))
	A_app = np.vstack((A_app, np.hstack((np.zeros((1, len(b))), np.array([[1]]))) ))

	########### Plot flow field with this appended matrix
	########### NOTE: setdimeq1 = True
	ffu.plot_flow(A_app, ax[0, 1], cmax = 10., width = .2, dim0 =0, dim1=1,
				setdimeq1 = True, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims)	
	
	#### Plot data 
	ax[0, 1].plot(X_tmp[:, 0], X_tmp[:, 1], 'k.-')
	ax[0, 1].plot(X_tmp[0, 0], X_tmp[0, 1], 'k.', markersize=20)
	ax[0, 1].set_title('Plot ff for [A|b; 0,1], \nsetdimeq1=True', fontsize=8)
	ax[0, 1].set_xlabel('X0')
	ax[0, 1].set_ylabel('X1')

	#### Plto fixed point estimate
	fixed_pt = np.dot(np.linalg.pinv(np.eye(2) - A), b)
	ax[0, 1].plot(fixed_pt[0], fixed_pt[1], 'r*')


	########## Plot in Z-space ##########
	### Get the eigenvalue / eigenvectors: 
	T, evs = ffu.get_sorted_realized_evs(A_app)
	T_inv = np.linalg.pinv(T)
	Za = np.real(np.dot(T_inv, np.dot(A_app, T)))

	#### compute z after appending ones to state: 
	X_tmp_app = np.hstack((X_tmp, np.ones((len(X_tmp), 1))))
	Z = np.dot(T_inv, X_tmp_app.T).T

	#### Plot flow field in z-space 
	#### Plot dimensions 1, 2 (not 0,1 )
	#### Dimension 0 is for the offset, so just a constant field 
	ffu.plot_flow(Za, ax[1, 1], cmax = 10., width = .2, dim0 =1, dim1=2,
				setdimeq1 = False, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims)	
	ax[1, 1].plot(Z[:, 1], Z[:, 2], 'k.-')
	ax[1, 1].plot(Z[0, 1], Z[0, 2], 'k.', markersize=20)
	ax[1, 1].set_title('Plot ff for Tinv*[A|b; 0,1]*T, \nsetdimeq1=False, dim1, dim2', fontsize=8)
	ax[1, 1].set_xlabel('Z1')
	ax[1, 1].set_ylabel('Z2')

	#### Here offset is zero  (z_{t+1} = Az*z_t, solution is z_t=z_{t+1} = 0)
	ax[1, 1].plot(0, 0, 'r*')
	
	######################## Method 2 ##########################################
	######################## Plot flow field from A, b separated #################
	
	########## Plot in X-space ##########
	#### Plot flow field from A, b separate
	#### Note: pls_b = b; 
	ffu.plot_flow(A, ax[0, 2], cmax = 10., width = .2, dim0 =0, dim1=1,
				setdimeq1 = False, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims, pls_b=b)	
	ax[0, 2].plot(X_tmp[:, 0], X_tmp[:, 1], 'k.-')
	ax[0, 2].plot(X_tmp[0, 0], X_tmp[0, 1], 'k.', markersize=20)
	ax[0, 2].set_title('Plot ff for A,b sep, \nsetdimeq1=False, pls_b=True', fontsize=8)
	ax[0, 2].set_xlabel('X0')
	ax[0, 2].set_ylabel('X1')

	###### Fixed pt: ######
	fixed_pt = np.dot(np.linalg.pinv(np.eye(2) - A), b)
	ax[0, 2].plot(fixed_pt[0], fixed_pt[1], 'r*')

	########## Plot in Z-space ##########
	T, evs = ffu.get_sorted_realized_evs(A)
	T_inv = np.linalg.pinv(T)
	Za = np.real(np.dot(T_inv, np.dot(A, T)))
	Zb = np.dot(T_inv, b)
	Z = np.dot(T_inv, X_tmp.T).T

	####### Plot z-space: note pls_b = Zb ########
	ffu.plot_flow(Za, ax[1, 2], cmax = 10., width = .2, dim0 =0, dim1=1,
				setdimeq1 = False, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims, pls_b = Zb)	
	ax[1, 2].plot(Z[:, 0], Z[:, 1], 'k.-')
	ax[1, 2].plot(Z[0, 0], Z[0, 1], 'k.', markersize=20)
	ax[1, 2].set_title('Plot ff for Tinv*A*T, Tinv*b sep, \nsetdimeq1=False, pls_b=True', fontsize=8)
	ax[1, 2].set_xlabel('Z0')
	ax[1, 2].set_ylabel('Z1')

	##### Fixed plt #############
	fixed_pt = np.dot(np.linalg.pinv(np.eye(2) - Za), Zb)
	ax[1, 2].plot(fixed_pt[0], fixed_pt[1], 'r*')

	f.tight_layout()