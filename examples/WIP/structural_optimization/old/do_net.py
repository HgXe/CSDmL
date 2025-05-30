# https://arxiv.org/abs/1907.04502
# https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html



"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import csdl_alpha as csdl

rec = csdl.Recorder()
rec.start()

# Load dataset
h5fname = 'struct_opt_aero_data_04.hdf5'
txtfname = 'pressure_eval_pts_2.txt'


pressure_eval_pts = np.loadtxt(txtfname)

geo_coeffs = []
pressure_vals = []
num_pts = 1000
for i in range(num_pts):
    var_dict = csdl.inline_import(h5fname, f'sample_{i}')
    geo_coeffs.append(var_dict['mono_wing_oml_coefficients'].value)
    pressure_vals.append(var_dict['pressure_eval'].value)

pressure_vals = np.array(pressure_vals)*1e-3
geo_coeffs = np.array(geo_coeffs)
geo_coeffs = geo_coeffs.reshape(geo_coeffs.shape[0], -1)


# select random points for training and testing
np.random.seed(0)

n_test_geo = 100
n_test_eval = 25

test_geo_indices = np.random.choice(geo_coeffs.shape[0], n_test_geo, replace=False)
test_eval_indices = np.random.choice(pressure_eval_pts.shape[0], n_test_eval, replace=False)

train_geo_indices = np.setdiff1d(np.arange(geo_coeffs.shape[0]), test_geo_indices)
train_eval_indices = np.setdiff1d(np.arange(pressure_eval_pts.shape[0]), test_eval_indices)

X_train = (geo_coeffs[train_geo_indices].astype(np.float32), pressure_eval_pts[train_eval_indices].astype(np.float32))
y_train = pressure_vals[train_geo_indices, :][:, train_eval_indices].astype(np.float32)

X_test = (geo_coeffs[test_geo_indices].astype(np.float32), pressure_eval_pts[test_eval_indices].astype(np.float32))
y_test = pressure_vals[test_geo_indices, :][:, test_eval_indices].astype(np.float32)

print('X_train:', X_train[0].shape, X_train[1].shape)
print('y_train:', y_train.shape)
print('X_test:', X_test[0].shape, X_test[1].shape)
print('y_test:', y_test.shape)




# d = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
# X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
# y_train = d["y"].astype(np.float32)
# d = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
# X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
# y_test = d["y"].astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# data = dde.data.Triple(
#     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
# )

# Choose a network
m = geo_coeffs.shape[1]
dim_x = pressure_eval_pts.shape[1]
net = dde.nn.DeepONetCartesianProd(
    [m, 1024, 1024, 1024, 512, 128],
    [dim_x, 512, 512, 128],
    "relu",
    "Glorot normal",
)
# net = dde.nn.DeepONet(
#     [m, 1024, 1024, 1024, 512, 128],
#     [dim_x, 128, 128, 128],
#     "relu",
#     "Glorot normal",
# )

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.00001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=100000)

# 10x the number of parameters is the data is standard
# Synergy with distributed optimization problem (branch and trunk)
# Think about evaluated -> evaluated with "reverse trunk?" - constrained lsq similar to Seb's Aeroelastic work conservation paper



# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()

_K = csdl.Variable()

_K = _K.set(csdl.slice[2*i:2*i+4, 2*i:2*i+4], _K[2*i:2*i+4, 2*i:2*i+4] + (_Ki_wo_x3 * x[i] ** 3))