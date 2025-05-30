from nets import Net1
import csdl_alpha as csdl
import numpy as np
import h5py
import optax

# inputs - edit the below
pressure_dim = 800
geo_dim = 117
disp_dim = 400

train_file_name = 'train_data.hdf5'
test_file_name = 'test_data.hdf5'
train_n = 800
test_n = 200
keys = ['pressure_coefficients', 
        'mono_wing_oml_coefficients',
        'displacement_coefficients']

hidden_dims = [100, 100, 100, 100]
activation = 'approx_relu'
loss_function = 'mse'

optimizer = optax.adam(1e-3)
num_batches = 10
num_epochs = 100
device = None
# end of inputs

dims = [pressure_dim, geo_dim*3, disp_dim*3]

# load data from file
Cp = np.zeros((train_n, dims[0]))
Cg = np.zeros((train_n, dims[1]))
Cd = np.zeros((train_n, dims[2]))

f = h5py.File(train_file_name, 'r')
# loop through the groups
for i in range(train_n):
    Cp[i] = f[f'sample_{i}'][keys[0]]
    Cg[i] = f[f'sample_{i}'][keys[1]]
    Cd[i] = f[f'sample_{i}'][keys[2]]
f.close()

# create the neural network
network = Net1(geo_dim=geo_dim, pressure_dim=pressure_dim, disp_dim=disp_dim, 
               hidden_dims=hidden_dims, activation=activation, loss_function=loss_function)

# train the network
network.train_jax_opt(optimizer=optimizer, loss_data=(Cg, Cp, Cd), num_batches=num_batches, 
                      num_epochs=num_epochs, device=device)