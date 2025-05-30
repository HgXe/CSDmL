import csdml as ml
import numpy as np
import csdl_alpha as csdl
from typing import Union, Callable

class Net1(ml.NeuralNetwork):
    '''
    Neural network for the first network in the structural optimization problem.
    
    Inputs: Cg, Cp
    Outputs: Cd

    Network diagram:
    Cg -> FCNN -> A 
                  v
                  x -> Cd   
    Cp -----------^
    '''
    def __init__(self, geo_dim:int, pressure_dim:int, disp_dim:int, hidden_dims:list[int],
                 activation:Union[Union[str, Callable], list[Union[str, Callable]]] = 'approx_relu', 
                 loss_function:Union[str, Callable] = 'mse'):
        """Initialize the Net1 neural network.

        Parameters
        ----------
        geo_dim : int
            The dimension of the geometric input features.
        pressure_dim : int
            The dimension of the pressure input features.
        disp_dim : int
            The dimension of the displacement output features.
        hidden_dims : list[int]
            A list of integers specifying the number of units in each hidden layer.
        activation : Union[Union[str, Callable], list[Union[str, Callable]]], optional
            The activation function(s) to use in the hidden layers. Can be a string specifying a known activation function,
            a callable for a custom activation function, or a list of such strings/callables for each layer. Default is 'approx_relu'.
        loss_function : Union[str, Callable], optional
            The loss function to use for training the network. Can be a string specifying a known loss function or a callable
            for a custom loss function. Default is 'mse'.
        """
        self.network = ml.FCNN(input_dim=geo_dim,
                               output_dim=pressure_dim*disp_dim,
                               hidden_dims=hidden_dims,
                               activation=activation,
                               loss_function=loss_function)
        self.pressure_dim = pressure_dim
        self.geo_dim = geo_dim
        self.disp_dim = disp_dim
        self.bias = csdl.Variable(value=np.random.randn(1))
        super().__init__(loss_function='mse', parameters=self.network.parameters+[self.bias])
        
    def init_parameters(self):
        self.network.init_parameters()
        self.bias = csdl.Variable(value=np.random.randn(1))
        self.parameters = self.network.parameters + [self.bias]

    def forward(self, Cg, Cp):
        A = self.network(Cg)
        if len(A.shape) == 2:
            A = A.reshape(A.shape[0], -1, self.pressure_dim)
        else:
            A = A.reshape(-1, self.pressure_dim)
        return A@Cp + self.bias
    
    def _forward(self, x):
        Cg = x[:, :-self.pressure_dim]
        Cp = x[:, -self.pressure_dim:]
        return self.forward(Cg, Cp)

    def train_jax_opt(self, optimizer:list, loss_data, num_batches=10, num_epochs=100, test_data=None, plot=True, device=None):
        Cg, Cp, Cd = loss_data
        X = np.concatenate([Cg, Cp], axis=-1)
        loss_data = (X, Cd)

        if test_data is not None:
            Cg_test, Cp_test, Cd_test = test_data
            X_test = np.concatenate([Cg_test, Cp_test], axis=-1)
            test_data = (X_test, Cd_test)

        return super().train_jax_opt(optimizer, loss_data, num_batches, num_epochs, test_data, plot, device)


