# theta = np.pi/4
# #A = np.array([[-np.sin(theta), np.cos(theta), 0],[np.cos(theta), np.sin(theta), 0], [0, 0, 1]])
# A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# A = np.array([[np.cos(theta), -np.sin(theta), 1],[np.sin(theta), np.cos(theta), 1], [0, 0, 1]])
import numpy as np; 
import analysis_config
import matplotlib.pyplot as plt 
import plot_flow_field_utils as ffu
from sklearn.linear_model import Ridge
pref_colors = ['maroon', 'orangered', 'darkgoldenrod', 'olivedrab', 
		'teal', 'steelblue', 'midnightblue', 'darkmagenta', 'black', 'gray']

# f, ax = plt.subplots()

# for i in range(10):
# 	x0 = np.random.randn(2,)
# 	x0 = np.hstack((x0, [1]))[:, np.newaxis]
	
# 	X = [x0]
# 	for _ in range(10):
# 		x1 = np.dot(A, x0)
# 		X.append(x1)
# 		x0 = x1.copy()

# 	X = np.hstack((X))
# 	ax.plot(X[0, :], X[1, :], '-',color = pref_colors[i])


def simulate_dyn(mean_sub = False, lims = 20):
	f, ax = plt.subplots(nrows = 4, ncols = 5, figsize = (10, 8))
	for i_d, decay in enumerate([.8]): #.8
		A_og = np.eye(2)*decay; 
		A_og[0, 0] = 0.75*A_og[0, 0]
		
		### Used later for estimating stable pt; 
		ImAinv = np.linalg.inv(np.eye(2) - A_og)

		for i_o, offs in enumerate(np.arange(4)):
			O = np.array([offs, offs])
			A = np.hstack((A_og, O[:, np.newaxis]))
			A = np.vstack((A, np.array([0., 0., 1])))

			ffu.plot_flow(A, ax[i_o, i_d], cmax = 8., width = .4, 
				setdimeq1 = True, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims,
				nb_points=10)

			### Compute the stable point ###
			stable_pt = np.squeeze(np.dot(ImAinv, np.array([offs, offs])[:, np.newaxis]))
			ax[i_o, i_d].plot(stable_pt[0], stable_pt[1], 'k.')
			ax[i_o, i_d].plot([-lims, lims], [0, 0], 'k--')
			ax[i_o, i_d].plot([0, 0], [-lims, lims], 'k--')
			ax[i_o, i_d].set_title('Decay=(%.1f, %.1f), Offs=%.1f\n Stable = (%.1f, %.1f)'%(decay, decay*.8, offs, stable_pt[0], stable_pt[1]))
			
			ev, _ = np.linalg.eig(A)
			plot_taus(ev, ax[i_o, i_d+1], offs = 3., color = 'k')

			#### Generate data from A; 
			X0 = []; X1 = [] 
			for i in range(25):
				for k in range(4):
					if k == 0:
						x0 = np.random.randn(2)*40
						x0 = np.hstack((x0, [1]))
						x0_og = x0.copy()
					elif k == 1:
						x0 = x0_og.copy()
						x0[0] *= -1
					elif k == 2:
						x0 = x0_og.copy()
						x0[1] *= -1
					elif k == 3:
						x0 = x0_og.copy()
						x0[:2] *= -1

					for j in range(10):
						x1 = np.squeeze(np.dot(A, x0[:, np.newaxis]))
						
						### Add 
						X0.append(x0[:2].copy())
						X1.append(x1[:2].copy())
						x0 = x1.copy()

			if mean_sub:
				mu = np.mean(np.vstack((X0)), axis=0)
				X0 = np.vstack((X0)) - mu[np.newaxis, :]
				X1 = np.vstack((X1)) - mu[np.newaxis, :]
				#mu = np.mean(np.vstack((X0)), axis=0)
			else:
				mu = np.mean(np.vstack((X0)), axis=0)
				X0 = np.vstack((X0))
				X1 = np.vstack((X1))

			cols = ['r','orangered', 'goldenrod','olivedrab','teal','steelblue','darkblue','purple', 'brown','black']
			for i in range(50):
				for ji, j in enumerate(np.arange(10)):
					tmp = X0[(i*10)+j, :]
					ax[i_o, i_d + 4].plot(tmp[0], tmp[1], '.',color=cols[ji], alpha=.5)
			ax[i_o, i_d+4].set_xlim([-lims, lims])
			ax[i_o, i_d+4].set_ylim([-lims, lims])

			### Day what the mean is; 
			ax[i_o, i_d+4].set_title('Mean: (%.1f, %.1f)' %(mu[0], mu[1]))
			
			ev_fit, A_fit = fit_ridge_get_tau(X0, X1, fit_intercept = True)
			ev_no_int, A_no_int = fit_ridge_get_tau(X0, X1, fit_intercept = False)

			### Here, get the intercept estimate; 
			B_est = np.squeeze(A_fit[:2, 2])
			ImAinv_est = np.linalg.inv(np.eye(2) - A_fit[np.ix_(np.arange(2), np.arange(2))])
			stable_pt_est = np.dot(ImAinv_est, np.squeeze(B_est)[:, np.newaxis])

			plot_taus(ev_fit, ax[i_o, i_d+1], offs = 2., color = 'b')
			plot_taus(ev_no_int, ax[i_o, i_d+1], offs = 1., color = 'r')

			ax[i_o, i_d+1].set_xlim([-150., 1500.])
			ax[i_o, i_d+1].set_title('K=true, B=Fit w int, R=Fit wo int', fontsize = 8)

			### Now plot what the dyn look like for offset vs. not; 
			ffu.plot_flow(A_fit, ax[i_o, i_d+2], cmax = 8., width = .12, 
				setdimeq1 = True, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims)	
			ax[i_o, i_d+2].set_title('Fit A w int\nStable Pt est = (%.1f, %.1f)' %(stable_pt_est[0], stable_pt_est[1]))

			ffu.plot_flow(A_no_int, ax[i_o, i_d+3], cmax = 8., width = .12, 
				setdimeq1 = False, xmin=-lims, xmax=lims, ymin=-lims, ymax=lims)	
			ax[i_o, i_d+3].set_title('Fit A no int')	
	f.tight_layout()	
	f.savefig(analysis_config.config['fig_dir']+'/sim_dyn.svg')

def plot_taus(evs, ax, offs = 1., color = 'k'):
	for e in evs:
		if e == 1.:
			pass
		else:

			dt = .1
			t = -1./np.log(np.abs(e))*dt*1000.
			ax.plot(t, offs, '.', color = color, markersize = 10)
			ax.plot([t, t], [0, offs], '-', color = color)

def fit_ridge_get_tau(X0, X1, dt = 0.1, fit_intercept = True):
	### Fit A w/o intercept; 
	clf = Ridge(alpha=0, fit_intercept=fit_intercept)
	clf.fit(np.vstack((X0)), np.vstack((X1)))
	A_fit = clf.coef_
	if fit_intercept:
		intc = clf.intercept_
		A_fit = np.hstack((A_fit, intc[:, np.newaxis]))
		A_fit = np.vstack((A_fit, np.array([0, 0, 1])))

	ev_fit, _ = np.linalg.eig(A_fit)
	return ev_fit, A_fit


def test_methods_of_ffplts(): 

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


