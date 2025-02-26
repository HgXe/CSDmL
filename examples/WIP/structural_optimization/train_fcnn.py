import csdml
import csdl_alpha as csdl
import h5py
import numpy as np
import jax
import optax

rec = csdl.Recorder()
rec.start()

# Load dataset
fname = 'struct_opt_geo_samples_01.hdf5'
num_samples = 800

group_names = {'displacement':'displacement_coefficients',
               'pressure':'pressure_coefficients',
               'geometry':'mono_wing_oml_coefficients'}
group_shapes = {'displacement':400*3, 'pressure':800, 'geometry':117*3}

Cd = np.zeros((num_samples, group_shapes['displacement']))
Cp = np.zeros((num_samples, group_shapes['pressure']))
Cg = np.zeros((num_samples, group_shapes['geometry']))


f = h5py.File(f'{fname}', 'r')
for i in range(num_samples):
    grp = f[f'sample_{i}']
    Cd[i] = grp[group_names['displacement']][...].flatten()
    Cp[i] = grp[group_names['pressure']][...].flatten()
    Cg[i] = grp[group_names['geometry']][...].flatten()
f.close()

# Normalize data
Cd_mean = np.mean(Cd, axis=0)
Cd_std = np.std(Cd, axis=0)
Cd = (Cd - Cd_mean) / Cd_std

Cp_mean = np.mean(Cp, axis=0)
Cp_std = np.std(Cp, axis=0)
Cp = (Cp - Cp_mean) / Cp_std

Cg_mean = np.mean(Cg, axis=0)
Cg_std = np.std(Cg, axis=0)
Cg = (Cg - Cg_mean) / Cg_std

# Split data
if True:
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:600]
    y_train = y[:600]

    X_test = X[600:]
    y_test = y[600:]

    # Train model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    hidden_dims = [64]
    epochs = 50000
    num_batches = 1
    dim_x = 1
    lr = 0.001
    # device = jax.devices('gpu')[0]
    device = jax.devices('cpu')[0]


    fcnn_net = csdml.FCNN(input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        output_dim=output_dim,
                        activation='tanh')

    optimizer = optax.adam(lr)
    loss_data = X_train, y_train
    test_data = X_test, y_test
    loss_history, test_loss_history, best_param_vals = fcnn_net.train_jax_opt(optimizer=optimizer, 
                                                                            loss_data=loss_data, 
                                                                            test_data=test_data, 
                                                                            num_batches=num_batches, 
                                                                            num_epochs=epochs, 
                                                                            device=device)

class SemiLinearNet(csdml.NeuralNetwork):
    def __init__(self, 
                 nl_input_dim:int, l_input_dim:int, hidden_dims:list[int], output_dim:int, 
                 activation = 'approx_relu', 
                 loss_function = 'mse'):
        
        self.nl_input_dim = nl_input_dim
        self.l_input_dim = l_input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation for _ in range(len(hidden_dims) + 1)]

        self.network = csdml.FCNN(input_dim=nl_input_dim,
                                  hidden_dims=hidden_dims,
                                  output_dim=output_dim*l_input_dim,
                                  activation=activation)

        super().__init__(loss_function, self.network.parameters)

    def init_parameters(self):
        self.network.init_parameters()
        self.parameters = self.network.parameters

    def forward(self, x1, x2):
        A = self.network(x1).reshape((-1, self.output_dim, self.l_input_dim))
        return csdl.einsum(A, x2, action='ijk,ik->ij')
    
    def _forward(self, x):
        x1 = x[:, :self.nl_input_dim]
        x2 = x[:, self.nl_input_dim:]
        return self.forward(x1, x2)
        







if False:
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:600]
    y_train = y[:600]

    X_test = X[600:]
    y_test = y[600:]

    # Train model
    nl_input_dim = Cg.shape[1]
    l_input_dim = Cp.shape[1]
    output_dim = y_train.shape[1]
    hidden_dims = [64]
    epochs = 500
    num_batches = 1
    dim_x = 1
    lr = 0.001
    # device = jax.devices('gpu')[0]
    device = jax.devices('cpu')[0]


    network = SemiLinearNet(nl_input_dim=nl_input_dim,
                            l_input_dim=l_input_dim,
                            hidden_dims=hidden_dims,
                            output_dim=output_dim,
                            activation='tanh')

    optimizer = optax.adam(lr)
    loss_data = X_train, y_train
    test_data = X_test, y_test
    loss_history, test_loss_history, best_param_vals = network.train_jax_opt(optimizer=optimizer, 
                                                                             loss_data=loss_data, 
                                                                             test_data=test_data, 
                                                                             num_batches=num_batches, 
                                                                             num_epochs=epochs,
                                                                             device=device)

