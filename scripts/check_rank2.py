from riglib.bmi import kfdecoder
from riglib.bmi import clda
from bmilist import bmi_state_space_models

ssm = bmi_state_space_models['Endpt2D']

update_rate = 0.1
A, _, W = ssm.get_ssm_matrices(update_rate=update_rate)

# C should be trained on all of the stochastic state variables, excluding the offset terms
n_features = 50; 
T = 1000; 
C = np.zeros((n_features, ssm.n_states))

kin = np.random.randn(int(ssm.n_states-1), T)
neural_features = np.random.randn(n_features, T)
C[:, ssm.drives_obs_inds], Q = kfdecoder.KalmanFilter.MLE_obs_model(kin[ssm.train_inds, :], neural_features)
C[:, [0, 2]] = np.random.randn(n_features, 2)

kf = kfdecoder.KalmanFilter(A, W, C, Q, is_stochastic=ssm.is_stochastic)

units = np.zeros((n_features, 2))
units[:, 0] = np.arange(n_features)

R, S, T, ESS = clda.KFRML.compute_suff_stats(kin, neural_features)
kf.R = R
kf.S = S
kf.T = T
kf.ESS = ESS

kd = kfdecoder.KFDecoder(kf, units, ssm, binlen=update_rate, tslice=(0, update_rate*T))
kd.n_features = n_features
F, K = kd.filt.get_sskf()
print('Decoder init -- allclose? %s'%np.allclose(K[3, :], 10*K[0, :]))
print('Decoder init -- rank 2 ? Rank: %s'%np.linalg.matrix_rank(K))
print('Decoder init -- rank C -- 2 ? Rank: %s'%np.linalg.matrix_rank(C))

### CLDA update
batch_time = 0.1
half_life = 300 
KFRML = clda.KFRML(batch_time, half_life) 
KFRML.init(kd)

params = KFRML.calc(np.random.randn(ssm.n_states, 1), np.random.randn(n_features, 1), kd)
kd.update_params(params)
F, K = kd.filt.get_sskf()
print('Decoder updated -- allclose? %s'%np.allclose(K[3, :], 10*K[0, :]))
print('Decoder updated -- rank 2 ? Rank: %s'%np.linalg.matrix_rank(K))
print('Decoder updated -- rank C -- 2 ? Rank: %s'%np.linalg.matrix_rank(kd.filt.C))

