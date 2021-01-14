
ncursor = 2
nneurons = 4
ninputs = 4

xdim = nneurons + ncursor; 
A = np.eye(xdim) # persistent dynamics in cursor/neural state

## Need some mapping betweenneurons and cursor state
### cursor is readout of neuron 1 and 2 
A[-ncursor:, :ncursor] = np.eye(ncursor) 

B = np.zeros((xdim, ninputs)) # map from nneurons to state
B[:nneurons, :nneurons] = np.eye(nneurons); # no direct inptu to the cursor, each input is to a neuron 

### Penalty on final state -- only for cursor 
Q_f = np.zeros((xdim, xdim))
Q_f[-ncursor:, -ncursor:] = np.eye(ncursor) 

### Penalty on input: 
R = np.eye(ninputs)

### Penalty on state during movement: 
Q = np.zeros((xdim, xdim))

A = np.mat(A)
B = np.mat(B)
Q_f = np.mat(Q_f)
Q = np.mat(Q)
R = np.mat(R)

T = 10
K = [None]*T
P = Q_f
for t in range(0,T-1)[::-1]:
    K[t] = (R + B.T*P*B).I * B.T*P*A
    P = Q + A.T*P*A -A.T*P*B*K[t]