import csdml
import csdl_alpha as csdl
import h5py
import numpy as np
import jax
import optax
import pickle
from viz import PlottingUtil

# def load_data(fname, num_samples, group_names, group_shapes, normalization_keys=[]):
#     arrays = {}
#     for key, shape in group_shapes.items():
#         arrays[key] = np.zeros((num_samples, shape)) 

#     f = h5py.File(f'{fname}', 'r')
#     for i in range(num_samples):
#         grp = f[f'sample_{i}']
#         for key in group_names.keys():
#             arrays[key][i] = grp[group_names[key]][...].flatten()
#     f.close()

#     # Normalize data to [-1, 1]
#     normalization_params = {}
#     for key in normalization_keys:
#         aarr_min = np.min(arrays[key], axis=0)
#         aarr_max = np.max(arrays[key], axis=0)
#         arrays[key] = 2*(arrays[key] - aarr_min) / (aarr_max - aarr_min) - 1
#         normalization_params[key] = {'min':aarr_min, 'max':aarr_max}

#     return arrays, normalization_params

def load_data(fnames, num_samples, group_names, group_shapes, normalization_keys=[], outlier_key=None):
    arrays = {key: [] for key in group_names.keys()}
    for _ in fnames:
        for key, shape in group_shapes.items():
            arrays[key].append(np.zeros((num_samples, shape))) 

    for i, fname in enumerate(fnames):
        f = h5py.File(f'{fname}', 'r')
        for j in range(num_samples):
            grp = f[f'sample_{j}']
            for key in group_names.keys():
                arrays[key][i][j] = grp[group_names[key]][...].flatten()
        f.close()

    # Concatenate arrays
    for key in group_names.keys():
        arrays[key] = np.concatenate(arrays[key], axis=0)

    if outlier_key is not None:
        # find indices of outliers
        m = 2.
        data = np.max(np.abs(arrays[outlier_key]), axis=1)
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        bad_inds = np.where(s>m)[0]
        print(bad_inds.shape[0], 'outliers found')

        # remove outliers from all data
        for key, array in arrays.items():
            arrays[key] = np.delete(array, bad_inds, axis=0)

    # Normalize by mean of data
    normalization_params = {}
    for key in normalization_keys:
        mean = np.mean(np.abs(arrays[key]))
        arrays[key] = arrays[key]/mean
        normalization_params[key] = {'mean':mean}


    return arrays, normalization_params

def train_fcnn(Cg, Cp, Cd, split_ind, net:csdml.FCNN, num_epochs):
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:split_ind]
    y_train = y[:split_ind]

    X_test = X[split_ind:]
    y_test = y[split_ind:]

    # Train model
    epochs = num_epochs
    num_batches = 1
    lr = 0.001
    device = jax.devices('gpu')[0]
    # device = jax.devices('cpu')[0]


    optimizer = optax.adam(lr)
    loss_data = X_train, y_train
    test_data = X_test, y_test
    loss_history, test_loss_history, best_param_vals = net.train_jax_opt(optimizer=optimizer, 
                                                                            loss_data=loss_data, 
                                                                            test_data=test_data, 
                                                                            num_batches=num_batches, 
                                                                            num_epochs=epochs, 
                                                                            device=device)
    
    return best_param_vals, loss_history, test_loss_history

def save_params(param_vals, fname):
    # pickle parameters
    with open(fname, 'wb') as f:
        pickle.dump(param_vals, f)

def load_params(fname, net:csdml.NeuralNetwork=None):
    # load parameters
    with open(fname, 'rb') as f:
        param_vals = pickle.load(f)
    if net is not None:
        net.set_param_values(param_vals)
    return param_vals




    # plot an example from the test set
    i = 0
    x = X_test[i]
    y = y_test[i]
    y_pred = fcnn_net(x)

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
        self.scale = csdl.Variable(value=np.random.randn(1))

        super().__init__(loss_function, self.network.parameters + [self.scale])

    def init_parameters(self):
        self.network.init_parameters()
        self.scale = csdl.Variable(value=np.random.randn(1))
        self.parameters = self.network.parameters + [self.scale]

    def forward(self, x1, x2):
        A = self.network(x1).reshape((-1, self.output_dim, self.l_input_dim))
        return csdl.einsum(A, x2, action='ijk,ik->ij') * self.scale
    
    def _forward(self, x):
        x1 = x[:, :self.nl_input_dim]
        x2 = x[:, self.nl_input_dim:]
        return self.forward(x1, x2)
        
class ReducedSLN(csdml.NeuralNetwork):
    def __init__(self, 
                 nl_input_dim:int, l_input_dim:int, hidden_dims:list[int], output_dim:int,
                 l_reduction_dim:int, nl_reduction_dim:int,
                 activation = 'approx_relu', 
                 loss_function = 'mse'):
        
        self.nl_input_dim = nl_input_dim
        self.l_input_dim = l_input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.l_reduction_dim = l_reduction_dim
        self.nl_reduction_dim = nl_reduction_dim
        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation for _ in range(len(hidden_dims) + 1)]

        self.network = csdml.FCNN(input_dim=nl_input_dim,
                                  hidden_dims=hidden_dims,
                                  output_dim=nl_reduction_dim*l_reduction_dim,
                                  activation=activation)
        self.scale = csdl.Variable(value=np.random.randn(1))
        self.l_reduction_matrix = csdl.Variable(value=np.random.randn(l_input_dim, l_reduction_dim))
        self.nl_reduction_matrix = csdl.Variable(value=np.random.randn(nl_reduction_dim, output_dim))

        super().__init__(loss_function, self.network.parameters + [self.scale, self.l_reduction_matrix, self.nl_reduction_matrix])

    def init_parameters(self):
        self.network.init_parameters()
        self.scale = csdl.Variable(value=np.random.randn(1))
        self.l_reduction_matrix = csdl.Variable(value=np.random.randn(self.l_input_dim, self.l_reduction_dim))
        self.nl_reduction_matrix = csdl.Variable(value=np.random.randn(self.nl_reduction_dim, self.output_dim))
        self.parameters = self.network.parameters + [self.scale, self.l_reduction_matrix, self.nl_reduction_matrix]

    def forward(self, x1, x2):
        A = self.network(x1).reshape((-1, self.nl_reduction_dim, self.l_reduction_dim))
        x2_r = x2 @ self.l_reduction_matrix
        y_r = csdl.einsum(A, x2_r, action='ijk,ik->ij') * self.scale
        y = y_r @ self.nl_reduction_matrix
        return y
    
    def _forward(self, x):
        x1 = x[:, :self.nl_input_dim]
        x2 = x[:, self.nl_input_dim:]
        return self.forward(x1, x2)

def train_semilinear(Cg, Cp, Cd, split_ind, net:csdml.NeuralNetwork, num_epochs, optimizer=optax.adam(0.001)):
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:split_ind]
    y_train = y[:split_ind]

    X_test = X[split_ind:]
    y_test = y[split_ind:]

    # Train model
    epochs = num_epochs
    num_batches = 1
    device = jax.devices('gpu')[0]
    # device = jax.devices('cpu')[0]
    loss_data = X_train, y_train
    test_data = X_test, y_test
    loss_history, test_loss_history, best_param_vals = net.train_jax_opt(optimizer=optimizer, 
                                                                             loss_data=loss_data, 
                                                                             test_data=test_data, 
                                                                             num_batches=num_batches, 
                                                                             num_epochs=epochs,
                                                                             device=device)
    return best_param_vals, loss_history, test_loss_history

rec = csdl.Recorder(inline=True)
rec.start()

do_training = True
do_training_optuna = False
num_epochs = 20000
# hidden_dims = [512, 512, 512, 256]
hidden_dims = [256, 256]


do_visualization = False

fnames = ['struct_opt_geo_with_nodes_01.hdf5', 'struct_opt_geo_with_nodes_02.hdf5', 'struct_opt_geo_with_nodes_03.hdf5', 'struct_opt_geo_with_nodes_04.hdf5']
test_fraction = 0.2

num_samples = 800
group_names = {'displacement':'displacement_coefficients',
            'pressure':'pressure_coefficients',
            'geometry':'mono_wing_oml_coefficients'}
group_shapes = {'displacement':400*3, 'pressure':800, 'geometry':117*3}

# Load dataset (& normalize)
arrays, normalization_params = load_data(fnames, num_samples, group_names, group_shapes, normalization_keys=['displacement', 'pressure', 'geometry'], outlier_key='displacement')
Cg = arrays['geometry']
Cp = arrays['pressure']
Cd = arrays['displacement']

split_ind = int(Cg.shape[0]*(1-test_fraction))

# net = csdml.FCNN(input_dim=Cg.shape[1] + Cp.shape[1],
#                      hidden_dims=hidden_dims,
#                      output_dim=Cd.shape[1],
#                      activation='tanh')

nl_net = SemiLinearNet(nl_input_dim=Cg.shape[1],
                        l_input_dim=Cp.shape[1],
                        hidden_dims=hidden_dims,
                        output_dim=Cd.shape[1],
                        activation='tanh')

rsl_net = ReducedSLN(nl_input_dim=Cg.shape[1],
                     l_input_dim=Cp.shape[1],
                     l_reduction_dim=16,
                     nl_reduction_dim=16,
                     hidden_dims=hidden_dims,
                     output_dim=Cd.shape[1],
                     activation='tanh')

if do_training:
    net = rsl_net

    schedule = optax.schedules.cosine_decay_schedule(0.003, num_epochs, alpha=0.0001)
    optimizer = optax.adam(0.001)

    with open('params_rsl_16x16_s2_aio.pkl', 'rb') as f:
        param_vals = pickle.load(f)
    net.set_param_values(param_vals)


    # params, loss, test_loss = train_fcnn(Cg, Cp, Cd, split_ind, net, num_epochs)
    params, loss, test_loss = train_semilinear(Cg, Cp, Cd, split_ind, net, num_epochs, optimizer=optimizer)

    save_params(params, 'params_rsl_16x16_s2_aio_2.pkl')

if do_training_optuna:
    import optuna

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-8, 0.003)
        optimizer = optax.adam(lr)
        params, loss, test_loss = train_semilinear(Cg, Cp, Cd, split_ind, rsl_net, num_epochs, optimizer=optimizer)
        return(min(test_loss))
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    best_lr = study.best_params['lr']

    print('best lr', best_lr)


if do_visualization:
    net = rsl_net

    load_params('params_rsl_16x16.pkl', net)

    # plot an example from the test set
    i = -10
    Cg_i = Cg[split_ind + i]
    Cp_i = Cp[split_ind + i]
    Cd_i = Cd[split_ind + i]
    # X = np.hstack((Cg_i, Cp_i)).reshape(1, -1)
    Cd_pred = net.forward(Cg_i.reshape(1,-1), Cp_i.reshape(1,-1)).flatten()

    # de-normalize
    # Cg_max = normalization_params['geometry']['max']
    # Cg_min = normalization_params['geometry']['min']
    # Cg_i = (Cg_i + 1) * (Cg_max - Cg_min) / 2 + Cg_min

    # Cd_max = normalization_params['displacement']['max']
    # Cd_min = normalization_params['displacement']['min']
    # Cd_i = (Cd_i + 1) * (Cd_max - Cd_min) / 2 + Cd_min
    # Cd_pred = (Cd_pred + 1) * (Cd_max - Cd_min) / 2 + Cd_min

    Cd_error = (Cd_i - Cd_pred)/Cd_i


    plotter = PlottingUtil()
    geo_function = plotter.make_geo_function(Cg_i.reshape(-1, 3))
    disp_error_function = plotter.make_disp_function(Cd_error.value.reshape(-1, 3))
    disp_function = plotter.make_disp_function(Cd_i.reshape(-1, 3))
    disp_pred_function = plotter.make_disp_function(Cd_pred.reshape(-1, 3))


    mesh = geo_function.plot_but_good(color=disp_pred_function, show=True)
    # import vedo
    # mesh.add_scalarbar()
    # plotter = vedo.Plotter()
    # plotter.show(mesh)


