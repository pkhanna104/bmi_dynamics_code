import numpy as np 
import scipy
import matplotlib.pyplot as plt 
from online_analysis import util_fcns

def flow_field_plot_top_dim(A, dt, X = None, dim0 = 0, dim1 = 1, cmax = .1,
    scale = 1.0, width = .04, ax = None):
    ''' 
    method to plot flow field plus first 100 data points in X after transforming 
    to the eigenvector basis 

    cmax, scale, and width and direct inputs to the plot_flow fcn
    '''

    assert(A.shape[0] == A.shape[1])
    
    if X is not None:
        assert(A.shape[0] == X.shape[1])

    ### Get the eigenvalue / eigenvectors: 
    T, evs = get_sorted_realized_evs(A)
    T_inv = np.linalg.pinv(T)

    ### Linear transform of A matrix
    Za = np.real(np.dot(T_inv, np.dot(A, T)))
    
    if X is not None:
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

    if X is None:
        Q = plot_flow(Za, ax, dim0=dim0, dim1=dim1, cmax = cmax, scale = scale, width = width)
    else:
        Q = plot_flow(Za, ax, dim0=dim0, dim1=dim1, xmax=xmax, xmin=xmin, ymax=ymax, 
          ymin=ymin, cmax = cmax, scale = scale, width = width)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        D = ax.plot(Z[:100, dim0], Z[:100, dim1], 'k.-', alpha=.5)
        D1 = ax.plot(Z[0, dim0], Z[0, dim1], 'r.', markersize=20)
    
    ax.set_xlabel('$z_{%d}$'%dim0, fontsize=14)
    ax.set_ylabel('$z_{%d}$'%dim1, fontsize=14)
    title_str = '$\lambda_{%d}$ Time Decay =%.2f sec, Hz = %.2f,\n $\lambda_{%d}$ Time Decay=%.2f sec, Hz = %.2f'%(dim0, td[0], hz0, dim1, td[1], hz1)
    ax.set_title(title_str, fontsize=14)
    if X is None:
        return Q
    else:
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
    scale = .5, alpha=1.0, width=.005, cmax=.1, setdimeq1 = False):

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
    DX, DY = compute_dX(X1, Y1, A, dim0, dim1, setdimeq1)  

    ### Get magnitude of difference
    M = (np.hypot(DX, DY))         

    ### Use quiver plot -- NOTE: axes must be "equal" to see arrows properly. 
    Q = axi.quiver(X1, Y1, DX, DY, M, units = 'xy', scale = scale,
        pivot='mid', cmap=plt.cm.viridis, width=width, alpha=alpha,
        clim = [0., cmax])
    return Q

def compute_dX(X, Y, A, dim0, dim1, setdimeq1):
    '''
    method to compute dX based on A
    '''
    if setdimeq1:
        non_dim = [i for i in range(len(A)) if i not in [dim0, dim1]]
        assert(len(non_dim) == 1)
        non_dim = non_dim[0]
        print('no zero dim: %d' %non_dim)
    else:
        non_dim = []

    newX = np.zeros_like(X)
    newY = np.zeros_like(Y)

    nrows, ncols = X.shape

    for nr in range(nrows):
        for nc in range(ncols):

            ### Assume everything is zero except dim1, dim2; 
            st = np.zeros((len(A), 1))
            st[non_dim] = 1.
            
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


def model_2_Amat(mod):
    ''' Add intercept onto the end and bottom of A matrix '''
    A = np.hstack(( mod.coef_, mod.intercept_[:, np.newaxis] ))
    _, n = A.shape
    A = np.vstack((A, np.hstack(( np.zeros((n-1, )), [1] )) ))
    return A

def plot_dyn_in_PC_space(model, pc_model, ax, cmax = 2., scale = 5.5, width = 0.01, lims = 2,
                   title = '', animal = 'grom', A = None):
    
    ### Add offset to A
    if A is None:
        A = model_2_Amat(model)

    ### Use the PCA model to make this work; 
    ### Neurons x nCPCs --> eigenvector projection; 
    trans_mat = pc_model['proj_mat']
    nn, npcs = trans_mat.shape

    ### Add offset terms; from PCs --> neurons 
    x_mn = pc_model['x_mn'][:, np.newaxis]
    trans_mat = np.hstack((trans_mat, x_mn)) ### N x (nPCs + 1)
    trans_mat = np.vstack((trans_mat, np.zeros((npcs+1)))) ## (N+1) x (nPCs + 1)
    trans_mat[-1, -1] = 1; # (N+1) x (nPCs + 1)
    
    ### Add offset terms; from neurons --> PCs
    trans_mat2 = pc_model['proj_mat'].T ### nPCs x Neurons 

    ### pc = U*(X-mu) = UX - Umu
    Uu = np.dot(trans_mat2, pc_model['x_mn'][:, np.newaxis]) ## nPcs
    trans_mat2 = np.hstack((trans_mat2, -1*Uu)) # nPCs x (Neurons + 1)
    trans_mat2 = np.vstack((trans_mat2, np.zeros((trans_mat2.shape[1], )))) ## (nPCs+1) x (NN+1)
    trans_mat2[-1, -1] = 1

    ########################################
    ################ Testing! ############
    N = 100
    test_data = np.random.randn(N, nn)
    pc_test_data = util_fcns.dat2PC(test_data, pc_model)
    
    ### This will only be true if # neurons == # PCs
    if npcs == nn:
        tmp = np.dot(trans_mat, np.hstack((pc_test_data, np.ones((N, 1)) )).T).T
        assert(np.allclose(tmp[:, :-1], test_data))
    
    tmp = np.dot(trans_mat2, np.hstack((test_data, np.ones((N, 1)))).T).T
    assert(np.allclose(tmp[:, :npcs], pc_test_data))

    ########################################
    ########################################

    ### Make modified KG: 
    PC_Amat = np.dot(trans_mat2, np.dot(A, trans_mat))
    ev,_ = np.linalg.eig(A)
    ev2,_ = np.linalg.eig(PC_Amat)
    
    if npcs == nn:
        assert(np.allclose(np.unique(ev), np.unique(ev2)))

    ### Plot the dynamics; 
    ax.axis('equal')
    plot_flow(PC_Amat, ax, cmax = cmax, scale = scale, 
                                    width = width, xmin=-lims, xmax=lims, ymin=-lims,
                                   ymax = lims, dim0=0, dim1=1, nb_points = 40, 
                                   setdimeq1 = True)
    ax.vlines(0, -lims, lims, 'k')
    ax.hlines(0, -lims, lims, 'k')