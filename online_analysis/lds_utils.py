#'3.6.9 (default, Nov  7 2019, 10:44:02) \n[GCC 8.3.0]'
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def plot_example_x0(A, nEx = 10, nT = 100, x1lim = (-5, 5), x2lim = (-5, 5), title = ''):
    ''' 
    function to take a given A matrix, randomly generate initial condition points in 
        2D space, and plot the resultant 2D trajectories when initial condition is propagated by A

    inputs: 
        A -- 2 x 2 dynamics matrix 
        nEx -- number of examples per x/y axis. Total number of examples = nEx**2
        nT -- number of time steps to propogate;  
    ''' 
    try:
        assert(A.shape[0] == A.shape[1] == 2)
    except:
        raise Exception('A must be a 2 x 2 matrix, this is a %d x %d matrix '%(A.shape[0], A.shape[1]))

    ### Generate nEx^2 examples of initial conditions, and propogate them nT steps; 
    ### Initial conditions grid + some noise to make visualization easier; 
    assert(x1lim[1] > x1lim[0])
    assert(x2lim[1] > x2lim[0])
    
    x1_init = np.linspace(x1lim[0], x1lim[1], nEx) + 0.1*np.random.randn(nEx)
    x2_init = np.linspace(x2lim[0], x2lim[1], nEx) + 0.1*np.random.randn(nEx)

    ### Setup the figure; 
    f, ax = plt.subplots(figsize = (5, 5))

    for i, xi1 in enumerate(x1_init):
        for j,xi2 in enumerate(x2_init):
            x0 = np.mat([[xi1], [xi2]])

            ### Save the trajectory ###
            x_traj = [x0]

            ### Generate the trajectory ###
            for it in range(nT):
                xi = np.dot(A, x0)
                x_traj.append(xi)
                x0 = xi.copy()
            
            ### Plot the trajectory 
            x_traj = np.hstack((x_traj)).T 
            assert(x_traj.shape[1] == 2)
            assert(x_traj.shape[0] == nT + 1)

            cax = ax.scatter(m2a(x_traj[:, 0]), m2a(x_traj[:, 1]), s=None, c=np.arange(nT+1), cmap = 'viridis')
    cbar = f.colorbar(cax)
    cbar.set_label('Timesteps', rotation = 270)
    ax.set_xlabel('$x^0$', fontsize=14)
    ax.set_ylabel('$x^1$', fontsize=14)
    ax.set_title(title, fontsize=14)

def eigenspec(*args, labels=None, dt = 0.01, xlim = None, ylim = None, axi = None, skip_legend=False):
    ''' 
    Method to plot the time decay (seconds) vs. frequency (hz) of eigenvalues of A for 
    each A in args
    '''

    if labels is not None:
        assert(len(labels) == len(args))
    
    ### Setup distinct colors for each A matrix
    N = len(args)
    colors = pl.cm.viridis(np.linspace(0,1,N))

    ### Setup plot
    if axi is None:
      f, ax = plt.subplots()
    else:
      ax = axi
      skip_legend = True

    L = []; 
    D = []; 
    for ia, A in enumerate(args):

        ### Make sure square matrix 
        assert(A.shape[0] == A.shape[1])

        ### Get eigenvalues: 
        ev, _ = np.linalg.eig(A)

        ### Get R / theta values
        R = np.abs(ev)
        Th = np.angle(ev)

        ## Get time decay
        TD = -1/np.log(R)*dt 

        ## Get Hz: 
        Hz = np.array(Th / (2*np.pi*dt))

        ix_mx = np.nonzero(np.round(Hz*1000)/1000.==(0.5/dt))[0]
        Hz[ix_mx] = 0

        if labels is None:
            lab = 'A%d'%(ia+1)
        else:
            lab = labels[ia]
        lines = ax.vlines(Hz, 0, TD, color=colors[ia])
        dots = ax.plot(Hz, TD, '.', color=colors[ia], label=lab, markersize=20)
        L.append(lines)
        D.append(dots)

    if skip_legend:
        pass
    else:
        f.legend()
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel('Time Decay (sec)')
    ax.set_xlabel('Frequency (Hz)')

    return L, D

###### PLOTTING TOP DIM #####
def flow_field_plot_top_dim(A, X, dt, dim0 = 0, dim1 = 1, cmax = .1,
    scale = 1.0, width = .04, ax = None):
    ''' 
    method to plot flow field plus first 100 data points in X after transforming 
    to the eigenvector basis 

    cmax, scale, and width and direct inputs to the plot_flow fcn
    '''

    assert(A.shape[0] == A.shape[1])
    assert(A.shape[0] == X.shape[1])
    nT, nD = X.shape

    ### Get the eigenvalue / eigenvectors: 
    T, evs = get_sorted_realized_evs(A)
    T_inv = np.linalg.pinv(T)

    ### Linear transform of A matrix
    Za = np.real(np.dot(T_inv, np.dot(A, T)))
    
    ### Transfrom the data; 
    Z = np.dot(T_inv, X.T).T

    xmax = np.max(np.real(Z[:100, dim0]))
    xmin = np.min(np.real(Z[:100, dim0]))
    ymax = np.max(np.real(Z[:100, dim1]))
    ymin = np.min(np.real(Z[:100, dim1]))

    ### Which eigenvalues are these adn what are their properties? 
    td = -1/np.log(np.abs(evs[[dim0, dim1]]))*dt; 
    hz0 = np.angle(evs[dim0])/(2*np.pi*dt)
    hz1 = np.angle(evs[dim1])/(2*np.pi*dt)

    ### Now plot flow field in top lambda dimensions
    if ax is None: 
        f, ax = plt.subplots()
    ax.axis('equal')
    Q = plot_flow(Za, ax, dim0=dim0, dim1=dim1, xmax=xmax, xmin=xmin, ymax=ymax, 
      ymin=ymin, cmax = cmax, scale = scale, width = width)

    D = ax.plot(Z[:100, dim0], Z[:100, dim1], 'k.-', alpha=.5)
    D1 = ax.plot(Z[0, dim0], Z[0, dim1], 'r.', markersize=20)
    ax.set_xlabel('$z_{%d}$'%dim0, fontsize=14)
    ax.set_ylabel('$z_{%d}$'%dim1, fontsize=14)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    title_str = '$\lambda_{%d}$ Time Decay =%.2f sec, Hz = %.2f,\n $\lambda_{%d}$ Time Decay=%.2f sec, Hz = %.2f'%(dim0, td[0], hz0, dim1, td[1], hz1)
    ax.set_title(title_str, fontsize=14)
    return Q, [D, D1]

def get_sorted_realized_evs(A):
    '''
    This method gives us the sorted eigenvalues/vectors that yields top dynamical dimensions
    as the first dimensions. For complex eigenvalues it also sets the first complex conjugate equal 
    to the real part and the second equal to the imaginary part. See https://www.youtube.com/watch?v=qlUr2Jc5O0g
    for more details; 
    ''' 

    ## Make sure A is a square; 
    assert(A.shape[0] == A.shape[1])
    
    ### Get eigenvalues / eigenvectors: 
    ev, evects = np.linalg.eig(A)

    ### Doesn't always give eigenvalues in order so sort them here: 
    ix_order = np.argsort(np.abs(ev)) ## This sorts them in increasing order; 
    ix_order_decreasing = ix_order[::-1]

    ### Sorted in decreasing order
    ev_sort = ev[ix_order_decreasing]
    evects_sort = evects[:, ix_order_decreasing]

    ### Make sure these eigenvectors/values still abide by Av = lv:
    chk_ev_vect(ev_sort, evects_sort, A)

    ### Now for imaginary eigenvalue, set the first part equal to the real, 
    ## and second part equal to the imaginary: 
    nD = A.shape[0]

    # Skip indices if complex conjugate
    skip_ix = [] 

    ## Go through each eigenvalue
    for i in range(nD):
        if i not in skip_ix:
            if np.imag(ev_sort[i]) != 0:
                evects_sort[:, i] = np.real(evects_sort[:, i])

                assert(np.real(ev_sort[i+1]) == np.real(ev_sort[i]))
                assert(np.imag(ev_sort[i+1]) == -1*np.imag(ev_sort[i]))

                evects_sort[:, i+1] = np.imag(evects_sort[:, i+1])
                skip_ix.append(i+1)

    return evects_sort, ev_sort
    

####### UTILS ######
def plot_flow(A, axi, nb_points=20, xmin=-5, xmax=5, ymin=-5, ymax=5, dim0 = 0, dim1 = 1,
    scale = .5, alpha=1.0, width=.005, cmax=.1):

    ''' Method to plot flow fields in 2D 
        Inputs: 
            A : an nxn matrix (datatype should be array)
            axi: axis on which to plot 
            nb_points: number of arrows to plot on x-axis adn y axis 
            x/y, min/max: limits of flow field plot 
            dim0: which dimension of A to plot on X axis; 
            dim1: which dimension of A to plot on Y axis; 
    '''

    x = np.linspace(xmin, xmax, nb_points)
    y = np.linspace(ymin, ymax, nb_points)
    # create a grid
    X1 , Y1  = np.meshgrid(x, y)                       
    
    ### For each position on the grid, (x1, y1), use A to compute where the next 
    ### point would be if propogate (x1, y1) by A -- assuming all other dimensions are zeros
    DX, DY = compute_dX(X1, Y1, A, dim0, dim1)  

    ### Get magnitude of difference
    M = (np.hypot(DX, DY))         

    ### Use quiver plot -- NOTE: axes must be "equal" to see arrows properly. 
    Q = axi.quiver(X1, Y1, DX, DY, M, units = 'xy', scale = scale,
        pivot='mid', cmap=plt.cm.viridis, width=width, alpha=alpha,
        clim = [0., cmax])
    return Q

def compute_dX(X, Y, A, dim0, dim1):
    '''
    method to compute dX based on A
    '''

    newX = np.zeros_like(X)
    newY = np.zeros_like(Y)

    nrows, ncols = X.shape

    for nr in range(nrows):
        for nc in range(ncols):

            ### Assume everything is zero except dim1, dim2; 
            st = np.zeros((len(A), 1))
            st[dim0] = X[nr, nc]; 
            st[dim1] = Y[nr, nc];

            st_nx = np.dot(A, st)
            newX[nr, nc] = st_nx[dim0]
            newY[nr, nc] = st_nx[dim1]

    ### Now to get the change, do new - old: 
    DX = newX - X; 
    DY = newY - Y; 

    return DX, DY

def chk_ev_vect(ev, evect, A):
    '''
    Check that each eigenvlaue / vecotr is correct: 
    '''
    for i in range(len(ev)):
        evi = ev[i]
        vct = evect[:, i]
        ## Do out the multiplication
        assert(np.allclose(np.dot(A, vct[:, np.newaxis]), evi*vct[:, np.newaxis]))

def m2a(x):
  '''
  method to squeeze 1D matrix into array format for easier plotting
  '''
  return np.squeeze(np.array(x))

def get_population_R2(X, X_est):
    ''' 
    get the variance in X accounted for in X_est 
    '''
    assert(X.shape == X_est.shape)

    ### Sum squared of residuals
    SSR = np.sum((X - X_est)**2)

    ### Sum squared total: 
    SST = np.sum((X - np.mean(X.reshape(-1)))**2)

    return 1 - SSR/SST