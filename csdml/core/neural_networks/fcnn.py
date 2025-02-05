import csdl_alpha as csdl
import numpy as np
from csdml.core.activation_functions import softplus, parametric_relu, relu_approximate
from csdml.core.neural_networks.neural_net import NeuralNetwork
from typing import Union, Callable
from csdl_alpha.utils.typing import VariableLike


class FCNN(NeuralNetwork):
    def __init__(self, 
                 input_dim:int, hidden_dims:list[int], output_dim:int, 
                 activation:Union[Union[str, Callable], list[Union[str, Callable]]] = 'approx_relu', 
                 loss_function:Union[str, Callable] = 'mse'):
        """
        Initialize a Fully Connected Neural Network (FCNN).

        Parameters
        ----------
        input_dim : int
            The dimension of the input layer.
        hidden_dims : list of int
            A list containing the dimensions of the hidden layers.
        output_dim : int
            The dimension of the output layer.
        activation : str, callable, or list of str or callable, optional
            The activation function to use for each layer. If a single string or callable is provided, 
            the same activation function will be used for all layers. If a list of strings or callables 
            is provided, each activation function will be used for the corresponding layer.
            Supported activation functions are 'relu', 'softplus', 'approx_relu', 'tanh', 'max', and 'linear'.
            Default is 'approx_relu'.
        loss_function : str or callable, optional
            The loss function to use. Can be 'mse' for mean squared error or a custom 
            callable function with signature loss(self, x_in, y_true, y_pred).  
            Default is 'mse'.
        """


        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        if isinstance(activation, list):
            self.activation = activation
        else:
            self.activation = [activation for _ in range(len(hidden_dims) + 1)]
        self.layers = []
        self.init_layers()
        self.init_parameters()
        super().__init__(loss_function, self.weights + self.biases)

    def init_parameters(self):
        self.init_weights()
        self.init_biases()
        self.parameters = self.weights + self.biases

    def init_layers(self):
        self.layers.append(self.input_dim)
        self.layers.extend(self.hidden_dims)
        self.layers.append(self.output_dim)

    def init_weights(self):
        self.weights = []
        for i in range(len(self.layers) - 1):
            weights_i = csdl.Variable(value=np.random.randn(self.layers[i], self.layers[i+1]))
            self.weights.append(weights_i)

    def init_biases(self):
        self.biases = []
        for i in range(len(self.layers) - 1):
            biases_i = csdl.Variable(value=np.random.randn(self.layers[i+1]))
            self.biases.append(biases_i)

    def forward(self, x:VariableLike) -> csdl.Variable:
        """
        Forward pass through the neural network.

        Parameters
        ----------
        x : csdl.VariableLike
            Input data. Vector or matrix with leading dimension being the batch size.
        
        Returns
        -------
        csdl.Variable
            Output of the neural network
        """
        mult = x.shape[0]
        for i in range(len(self.layers) - 1):
            # print(x.shape, self.weights[i].shape, self.biases[i].shape)
            bias = csdl.expand(self.biases[i], out_shape=(mult,self.biases[i].shape[0]), action='i->ji')
            x = x @ self.weights[i] + bias
            activation = self.activation[i]
            if activation == 'softplus':
                x = softplus(x)
            elif activation == 'tanh':
                x = csdl.tanh(x)
            elif activation == 'max':
                x = csdl.maximum(x)
            elif activation == 'relu':
                x = parametric_relu(x)
            elif activation == 'approx_relu':
                x = relu_approximate(x)
            elif activation is None or activation == 'linear':
                pass
            elif callable(activation):
                x = activation(x)
            else:
                raise ValueError(f'Invalid activation function for layer {i}')
        return x

def test_jax_opt():
    from jax.example_libraries import optimizers as jax_opt

    # test the FCNN class
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.rand(10000, 1)*2*np.pi
    y = np.sin(X)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
    Y_test = np.sin(X_test)

    activation = ['approx_relu', 'tanh', 'tanh', 'tanh', 'tanh']
    model = FCNN(1, [20, 20, 20, 20], 1, activation=activation)
    loss_data = X, y

    optimizer = jax_opt.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_epochs=1000)

    # # plot model and training function
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    
    # y_pred = model.forward(X_test).value
    # __=ax.plot(X_test, y_pred)
    # __=ax.plot(X_test, Y_test)

    # ax.legend(['Predicted', 'True'])

    # plt.show()

def test_optax_opt():
    import optax

    # test the FCNN class
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.rand(10000, 1)*2*np.pi
    y = np.sin(X)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
    Y_test = np.sin(X_test)

    activation = ['relu', 'tanh', 'tanh', 'tanh', 'tanh']
    model = FCNN(1, [20, 20, 20, 20], 1, activation=activation)
    loss_data = X, y

    optimizer = optax.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_epochs=1000)


    # # plot model and training function
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    
    # y_pred = model.forward(X_test).value
    # __=ax.plot(X_test, y_pred)
    # __=ax.plot(X_test, Y_test)

    # ax.legend(['Predicted', 'True'])

    # plt.show()

def test_custom_loss_optax():
    import optax
    from csdml import softplus

    # test the FCNN class
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.rand(1000, 1)
    y = np.sin(X*2*np.pi)

    # # plot the training data
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # __=ax.scatter(X, y)
    # plt.show()

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    Y_test = np.sin(X_test*2*np.pi)
    derivative_loss_factor = 0.1

    def custom_loss(net, x_in:csdl.Variable, y_true:csdl.Variable, y_pred:csdl.Variable):
        # add gradient penalty
        
        dy_dx_pred_matrix = csdl.derivative(y_pred, x_in, elementwise=True)
        dy_dx_pred = csdl.Variable(shape=y_pred.shape, value=0)
        for i in csdl.frange(dy_dx_pred_matrix.shape[0]):
            dy_dx_pred = dy_dx_pred.set(csdl.slice[i], dy_dx_pred_matrix[i,i])

        dy_dx_true = csdl.cos((2*np.pi)*x_in)*(2*np.pi)

        forward_loss = csdl.norm(y_pred - y_true)/x_in.shape[0]
        derivative_loss = csdl.norm(dy_dx_pred - dy_dx_true)/x_in.shape[0]
        return (1 - derivative_loss_factor)*forward_loss + derivative_loss_factor*derivative_loss


    prelu = lambda x: parametric_relu(x, alpha=0.1)
    activation = [prelu, 'tanh']
    model = FCNN(1, [100], 1, activation=activation, loss_function=custom_loss)
    loss_data = X, y

    optimizer = optax.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_batches=1, num_epochs=1000)


    # # plot model and training function
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    
    # X_test_var = csdl.Variable(value=X_test)
    # y_pred = model.forward(X_test_var)
    # y_pred_val = y_pred.value
    # __=ax.plot(X_test, y_pred_val)
    # __=ax.plot(X_test, Y_test)

    # ax.legend(['Predicted', 'True'])

    # plt.show()

    # # plot derivative of the model and the true derivative
    # fig, ax = plt.subplots(1, 1)
    # dy_pred = np.diag(csdl.derivative(y_pred, X_test_var, elementwise=True).value)
    # dy_true = np.cos(X_test*2*np.pi)*2*np.pi
    # __=ax.plot(X_test, dy_pred)
    # __=ax.plot(X_test, dy_true)

    # ax.legend(['Predicted', 'True'])

    # plt.show()

def test_custom_loss_jaxopt():
    from jax.example_libraries import optimizers as jax_opt

    # test the FCNN class
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.rand(1000, 1)
    y = np.sin(X*2*np.pi)

    # # plot the training data
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # __=ax.scatter(X, y)
    # plt.show()

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    Y_test = np.sin(X_test*2*np.pi)
    derivative_loss_factor = 0.1

    def custom_loss(net, x_in:csdl.Variable, y_true:csdl.Variable, y_pred:csdl.Variable):
        # add gradient penalty
        
        dy_dx_pred_matrix = csdl.derivative(y_pred, x_in, elementwise=True)
        dy_dx_pred = csdl.Variable(shape=y_pred.shape, value=0)
        for i in csdl.frange(dy_dx_pred_matrix.shape[0]):
            dy_dx_pred = dy_dx_pred.set(csdl.slice[i], dy_dx_pred_matrix[i,i])

        dy_dx_true = csdl.cos((2*np.pi)*x_in)*(2*np.pi)

        forward_loss = csdl.norm(y_pred - y_true)/x_in.shape[0]
        derivative_loss = csdl.norm(dy_dx_pred - dy_dx_true)/x_in.shape[0]
        return (1 - derivative_loss_factor)*forward_loss + derivative_loss_factor*derivative_loss


    prelu = lambda x: parametric_relu(x, alpha=0.1)
    activation = [prelu, 'tanh']
    model = FCNN(1, [100], 1, activation=activation, loss_function=custom_loss)
    loss_data = X, y

    optimizer = jax_opt.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_batches=1, num_epochs=1000)


    # # plot model and training function
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    
    # X_test_var = csdl.Variable(value=X_test)
    # y_pred = model.forward(X_test_var)
    # y_pred_val = y_pred.value
    # __=ax.plot(X_test, y_pred_val)
    # __=ax.plot(X_test, Y_test)

    # ax.legend(['Predicted', 'True'])

    # plt.show()

    # # plot derivative of the model and the true derivative
    # fig, ax = plt.subplots(1, 1)
    # dy_pred = np.diag(csdl.derivative(y_pred, X_test_var, elementwise=True).value)
    # dy_true = np.cos(X_test*2*np.pi)*2*np.pi
    # __=ax.plot(X_test, dy_pred)
    # __=ax.plot(X_test, dy_true)

    # ax.legend(['Predicted', 'True'])

    # plt.show()


if __name__ == '__main__':
    # test_jax_opt()
    # test_optax_opt()
    # test_custom_loss_optax()
    test_custom_loss_jaxopt()

