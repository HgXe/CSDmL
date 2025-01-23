import csdl_alpha as csdl
import numpy as np
from csdml.core.activation_functions import gelu




class FCNN():
    def __init__(self, input_dim, hidden_dims, output_dim, activation = 'gelu', loss_function = 'mse'):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.layers = []
        self.init_layers()
        self.init_weights()
        self.init_biases()
        self.loss_function = loss_function

    def init_layers(self):
        self.layers.append(self.input_dim)
        self.layers.extend(self.hidden_dims)
        self.layers.append(self.output_dim)

    def init_weights(self, value:list[csdl.Variable]=None):
        self.weights = []
        for i in range(len(self.layers) - 1):
            if value is not None:
                weights_i = csdl.Variable(shape=(self.layers[i], self.layers[i+1]), value=value[i])
            else:
                weights_i = csdl.Variable(value=np.random.randn(self.layers[i], self.layers[i+1]))
            self.weights.append(weights_i)

    def init_biases(self, value:list[csdl.Variable]=None):
        self.biases = []
        for i in range(len(self.layers) - 1):
            if value is not None:
                biases_i = csdl.Variable(shape=(self.layers[i+1],), value=value[i])
            else:
                biases_i = csdl.Variable(value=np.random.randn(self.layers[i+1]))
            self.biases.append(biases_i)

    def forward(self, x):
        mult = x.shape[0]
        for i in range(len(self.layers) - 1):
            # print(x.shape, self.weights[i].shape, self.biases[i].shape)
            bias = csdl.expand(self.biases[i], out_shape=(mult,self.biases[i].shape[0]), action='i->ji')
            x = x @ self.weights[i] + bias
            if self.activation == 'gelu':
                x = gelu(x)
            elif self.activation == 'tanh':
                x = csdl.tanh(x)
            else:
                raise ValueError('Invalid activation function')
        return x
    
    def __call__(self, x):
        return self.forward(x)

    def set_design_variables(self):
        self.init_weights()
        self.init_biases()

        # set design variables (weights and biases) to be optimized
        for weight, bias in zip(self.weights, self.biases):
            weight.set_as_design_variable()
            bias.set_as_design_variable()

    def set_optimized_values(self):
        # set the weights and biases to the optimized values
        self.init_weights([weight.value for weight in self.weights])
        self.init_biases([bias.value for bias in self.biases])

    def train(self, X, y):
        rec_outer = csdl.get_current_recorder()
        rec_outer.stop()

        # create a new recorder that only records the neural network
        rec_inner = csdl.Recorder()
        rec_inner.start()

        self.set_design_variables

        # run the training loop
        y_pred = self.forward(X)

        # compute the loss
        if self.loss_function == 'mse':
            loss = csdl.norm((y - y_pred))
        elif callable(self.loss_function):
            loss = self.loss_function(y, y_pred)
        else:
            raise ValueError('Invalid loss function')
        loss.set_as_objective()

        # optimize the design variables
        import modopt
        sim = csdl.experimental.JaxSimulator(rec_inner)
        problem = modopt.CSDLAlphaProblem(problem_name='FCNN', simulator=sim)
        optimizer = modopt.PySLSQP(problem, solver_options={'maxiter': 1000})
        optimizer.solve()
        optimizer.print_results()
        
        # switch back to the outer recorder
        rec_inner.stop()
        rec_outer.start()

        self.set_optimized_values()
        


if __name__ == '__main__':
    # test the FCNN class
    rec = csdl.Recorder()
    rec.start()

    X = np.random.randn(100, 10)
    y = np.sin(np.sum(X, axis=1)).reshape(-1, 1)
    model = FCNN(10, [20, 20], 1)
    model.train(X, y)




# def correction_function(
#         x:csdl.Variable, #inputs as a vector
#         i:int,
#     ):
#     # activation_function = lambda x: csdl.maximum(np.zeros((num_neurons,)),x)
#     # activation_function = lambda x: csdl.maximum(0.1*x,x)
#     activation_function = lambda x: csdl.tanh(x)

#     for ii in range(4):
#         n = x.size
#         weights = csdl.Variable(name=f'w_{i}_{ii}',shape=(num_neurons,n), value=-10*(np.random.rand(num_neurons,n)-0.5))
#         bias = csdl.Variable(name=f'b_{i}_{ii}',shape=(num_neurons,), value=-10*(np.random.rand(num_neurons,)-0.5))
#         new_x = activation_function(weights@x+bias)
#         x = new_x

#     weights_out = csdl.Variable(name=f'wo_{i}',shape=(1,num_neurons,), value=-10.00*(np.random.rand(1,num_neurons)-0.5))
#     b_out = csdl.Variable(name=f'bo_{i}',shape=(1,), value=-0*(np.random.rand(1,)-0.5))

#     return weights_out@new_x+b_out