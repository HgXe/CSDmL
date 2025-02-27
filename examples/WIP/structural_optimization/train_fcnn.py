import csdml
import csdl_alpha as csdl
import h5py
import numpy as np
import jax
import optax
import pickle
from viz import PlottingUtil

def load_data(fname, num_samples, group_names, group_shapes, normalization_keys=[]):
    arrays = {}
    for key, shape in group_shapes.items():
        arrays[key] = np.zeros((num_samples, shape)) 

    f = h5py.File(f'{fname}', 'r')
    for i in range(num_samples):
        grp = f[f'sample_{i}']
        for key in group_names.keys():
            arrays[key][i] = grp[group_names[key]][...].flatten()
    f.close()

    # Normalize data to [-1, 1]
    normalization_params = {}
    for key in normalization_keys:
        aarr_min = np.min(arrays[key], axis=0)
        aarr_max = np.max(arrays[key], axis=0)
        arrays[key] = 2*(arrays[key] - aarr_min) / (aarr_max - aarr_min) - 1
        normalization_params[key] = {'min':aarr_min, 'max':aarr_max}

    return arrays, normalization_params


def train_fcnn(Cg, Cp, Cd, split_ind, net:csdml.FCNN):
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:split_ind]
    y_train = y[:split_ind]

    X_test = X[split_ind:]
    y_test = y[split_ind:]

    # Train model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    hidden_dims = [64]
    epochs = 5000
    num_batches = 1
    dim_x = 1
    lr = 0.001
    # device = jax.devices('gpu')[0]
    device = jax.devices('cpu')[0]


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
        
def train_semilinear(Cg, Cp, Cd, split_ind, net):
    X = np.hstack((Cg, Cp))
    y = Cd

    X_train = X[:split_ind]
    y_train = y[:split_ind]

    X_test = X[split_ind:]
    y_test = y[split_ind:]

    # Train model
    nl_input_dim = Cg.shape[1]
    l_input_dim = Cp.shape[1]
    output_dim = y_train.shape[1]
    hidden_dims = [64]
    epochs = 50
    num_batches = 1
    dim_x = 1
    lr = 0.001
    # device = jax.devices('gpu')[0]
    device = jax.devices('cpu')[0]




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

rec = csdl.Recorder(inline=True)
rec.start()

do_training = True
do_visualization = True

# Load dataset (& normalize)
fname = 'struct_opt_geo_samples_01.hdf5'
num_samples = 800

group_names = {'displacement':'displacement_coefficients',
            'pressure':'pressure_coefficients',
            'geometry':'mono_wing_oml_coefficients'}
group_shapes = {'displacement':400*3, 'pressure':800, 'geometry':117*3}

arrays, normalization_params = load_data(fname, num_samples, group_names, group_shapes, normalization_keys=['displacement'])
Cg = arrays['geometry']
Cp = arrays['pressure']
Cd = arrays['displacement']

net = csdml.FCNN(input_dim=Cg.shape[1] + Cp.shape[1],
                     hidden_dims=[512, 128],
                     output_dim=Cd.shape[1],
                     activation='tanh')

nl_net = SemiLinearNet(nl_input_dim=Cg.shape[1],
                        l_input_dim=Cp.shape[1],
                        hidden_dims=[64],
                        output_dim=Cd.shape[1],
                        activation='tanh')

if do_training:
    params, loss, test_loss = train_fcnn(Cg, Cp, Cd, 600, net)
    # params, loss, test_loss = train_semilinear(Cg, Cp, Cd, 600, nl_net)

    save_params(params, 'params.pkl')

if do_visualization:

    load_params('params.pkl', net)

    # plot an example from the test set
    i = -10
    Cg_i = Cg[600 + i]
    Cp_i = Cp[600 + i]
    Cd_i = Cd[600 + i]
    X = np.hstack((Cg_i, Cp_i)).reshape(1, -1)
    Cd_pred = net(X).flatten()

    # de-normalize
    # Cg_max = normalization_params['geometry']['max']
    # Cg_min = normalization_params['geometry']['min']
    # Cg_i = (Cg_i + 1) * (Cg_max - Cg_min) / 2 + Cg_min

    Cd_max = normalization_params['displacement']['max']
    Cd_min = normalization_params['displacement']['min']
    Cd_i = (Cd_i + 1) * (Cd_max - Cd_min) / 2 + Cd_min
    Cd_pred = (Cd_pred + 1) * (Cd_max - Cd_min) / 2 + Cd_min

    Cd_error = (Cd_i - Cd_pred)/Cd_i


    plotter = PlottingUtil()
    geo_function = plotter.make_geo_function(Cg_i.reshape(-1, 3))
    disp_error_function = plotter.make_disp_function(Cd_error.value.reshape(-1, 3))
    disp_function = plotter.make_disp_function(Cd_i.reshape(-1, 3))
    disp_pred_function = plotter.make_disp_function(Cd_pred.reshape(-1, 3))


    mesh = geo_function.plot_but_good(color=disp_error_function, show=True)
    # import vedo
    # mesh.add_scalarbar()
    # plotter = vedo.Plotter()
    # plotter.show(mesh)


