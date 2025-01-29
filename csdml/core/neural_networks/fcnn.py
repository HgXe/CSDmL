import csdl_alpha as csdl
import numpy as np
from csdml.core.activation_functions import gelu
from csdml.core.neural_networks.neural_net import NeuralNetwork


class FCNN(NeuralNetwork):
    def __init__(self, input_dim, hidden_dims, output_dim, activation = 'gelu', loss_function = 'mse'):
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

    def init_biases(self, value:list[csdl.Variable]=None):
        self.biases = []
        for i in range(len(self.layers) - 1):
            biases_i = csdl.Variable(value=np.random.randn(self.layers[i+1]))
            self.biases.append(biases_i)

    def forward(self, x):
        mult = x.shape[0]
        for i in range(len(self.layers) - 1):
            # print(x.shape, self.weights[i].shape, self.biases[i].shape)
            bias = csdl.expand(self.biases[i], out_shape=(mult,self.biases[i].shape[0]), action='i->ji')
            x = x @ self.weights[i] + bias
            activation = self.activation[i]
            if activation == 'gelu':
                x = gelu(x)
            elif activation == 'tanh':
                x = csdl.tanh(x)
            elif activation == 'max':
                x = csdl.maximum(x)
            elif activation is None or activation == 'linear':
                pass
            elif activation is callable:
                x = activation(x)
            else:
                raise ValueError(f'Invalid activation function for layer {i}')
        return x

if __name__ == '__main__':
    from jax.example_libraries import optimizers as jax_opt

    # test the FCNN class
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.randn(10000, 1)*2*np.pi
    y = np.sin(X)

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
    Y_test = np.sin(X_test)

    activation = ['gelu', 'tanh', 'tanh', 'tanh', 'tanh']
    model = FCNN(1, [20, 20, 20, 20], 1, activation=activation)
    loss_data = X, y

    optimizer = jax_opt.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_epochs=1000)
    # model.train_adam(loss_data, test_data=(X_test, Y_test), num_epochs=1000, step_size=1e-4)

    # plot model and training function
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    
    y_pred = model.forward(X_test).value
    __=ax.plot(X_test, y_pred)
    __=ax.plot(X_test, Y_test)

    ax.legend(['Predicted', 'True'])

    plt.show()

