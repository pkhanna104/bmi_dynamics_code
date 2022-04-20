from sklearn import linear_model

f, ax = plt.subplots(ncols=2)
for alpha in [0., 1., 10., 100., 1000., 10000.]:

	## If no intercept, then I have the equation correctly ####
	## If there is an intercept fit, then X and Y must be mean subtracted first to fit B
	## THen intercept is found after adding back intercepts 
	clf = linear_model.Ridge(alpha=alpha, fit_intercept=True)
	X = np.hstack(( np.arange(10)[:, np.newaxis], 0.1*np.random.randn(10, 1)))
	Y = 1000*X#[:, 0]
	#Y = Y[:, np.newaxis]

	X_offs = np.mean(X, axis=0)
	Y_offs = np.mean(Y, axis=0)

	#X = X - X_offs
	#Y = Y - Y_offs

	clf.fit(X, Y)
	if alpha == 0:
		xtmp = -1
	else:
		xtmp = np.log10(alpha)

	### estimated 
	X_new = X - X_offs[np.newaxis, :]
	Y_new = Y - Y_offs[np.newaxis, :]
	covX = np.dot(X_new.T, X_new)
	covXY = np.dot(X_new.T, Y_new)
	B = np.dot(covXY.T, np.linalg.pinv(alpha*np.eye(X.shape[1]) + covX))
	
	### Intercept is
	B0 = np.mean(Y - np.dot(B, X.T).T, axis=0)

	ax[0].plot(xtmp, clf.intercept_[0], 'k.')
	ax[0].plot(xtmp, B0[0], 'b.')

	ax[1].plot(xtmp, clf.coef_[0, 0], 'k.')
	ax[1].plot(xtmp, B[0, 0], 'b.')

ax[0].hlines(0, -1, 5, 'r')
ax[1].hlines(1000, -1, 5, 'r')