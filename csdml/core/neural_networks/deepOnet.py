import csdl_alpha as csdl
import numpy as np
from csdml.core.neural_networks.fcnn import FCNN
from csdml.core.neural_networks.neural_net import NeuralNetwork

class DeepOnet(NeuralNetwork):
    def __init__(self, trunk:FCNN, branch:FCNN, output_dim:int=1, loss_function='mse'):
        self.trunk = trunk
        self.branch = branch
        self.output_dim = output_dim
        super().__init__(loss_function, self.trunk.parameters + self.branch.parameters)

    def init_parameters(self):
        self.trunk.init_parameters()
        self.branch.init_parameters()
        self.parameters = self.trunk.parameters + self.branch.parameters

    def forward(self, x, t=None):
        
        # need this for it to work with the training loop
        if t is None:
            x = x[:, :-self.trunk.input_dim]
            t = x[:, -self.trunk.input_dim:]

        z = self.branch(x)
        y = self.trunk(t)

        # TODO: make outoput dim do something
        if self.output_dim != 1:
            raise NotImplementedError('Output dim other than 1 is not implemented')

        # TODO: consider adding bias here
        return csdl.einsum(z, y, action='ij,ij->i').reshape(-1, 1)


    def train_jax_opt(self, optimizer:list, loss_data, num_batches=10, num_epochs=100, test_data=None, plot=True, device=None):
        X, t, Y = loss_data
        X_t = np.concatenate([X, t], axis=-1)
        repackaged_loss_data = (X_t, Y)

        if test_data is not None:
            X_test, t_test, Y_test = test_data
            X2_test = np.concatenate([X_test, t_test], axis=-1)
            repackaged_test_data = (X2_test, Y_test)
        else:
            repackaged_test_data = None

        super().train_jax_opt(optimizer, repackaged_loss_data, num_batches, num_epochs, repackaged_test_data, plot, device)

    # def train(self, X, T, y):
    #     rec_outer = csdl.get_current_recorder()
    #     rec_outer.stop()

    #     # create a new recorder that only records the neural network
    #     rec_inner = csdl.Recorder()
    #     rec_inner.start()

    #     self.trunk.set_design_variables()
    #     self.branch.set_design_variables()

    #     y_pred = self.forward(X, T)

    #      # compute the loss
    #     if self.loss_function == 'mse':
    #         loss = csdl.norm((y - y_pred))
    #     elif callable(self.loss_function):
    #         loss = self.loss_function(y, y_pred)
    #     else:
    #         raise ValueError('Invalid loss function')
    #     loss.set_as_objective()

    #     # optimize the design variables
    #     import modopt
    #     sim = csdl.experimental.JaxSimulator(rec_inner)
    #     problem = modopt.CSDLAlphaProblem(problem_name='FCNN', simulator=sim)
    #     optimizer = modopt.PySLSQP(problem, solver_options={'maxiter': 1000})
    #     optimizer.solve()
    #     optimizer.print_results()
        
    #     # switch back to the outer recorder
    #     rec_inner.stop()
    #     rec_outer.start()

    #     self.trunk.set_optimized_values()
    #     self.branch.set_optimized_values()


if __name__ == '__main__':
    from jax.example_libraries import optimizers as jax_opt
    import jax

    # test the DeepOnet class
    rec = csdl.Recorder(inline=True)
    rec.start()

    # X = np.random.randn(10000, 100)*2*np.pi
    # T = np.random.randn(10000, 1)*2*np.pi
    # y = np.sin(X) * np.cos(T)

    # X_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
    # T_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
    # Y_test = np.sin(X_test) * np.cos(T_test)

    test_data = np.load('test.npz')
    X_test = test_data['X_test0']
    T_test = test_data['X_test1']
    Y_test = test_data['y_test']

    train_data = np.load('train.npz')
    X = train_data['X_train0']
    T = train_data['X_train1']
    y = train_data['y_train']


    m = 240
    epochs = 20000
    dim_x = 1
    lr = 0.001
    device = jax.devices('gpu')[0]

    branch = FCNN(m, [100], 100, activation='gelu')
    trunk = FCNN(dim_x, [100], 100, activation='gelu')

    model = DeepOnet(trunk, branch)
    loss_data = X, T, y


    optimizer = jax_opt.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, T_test, Y_test), num_epochs=epochs, device=device)


    test_data = np.load('test.npz')
    X_test = test_data['X_test0']
    T_test = test_data['X_test1']
    Y_test = test_data['y_test']

    train_data = np.load('train.npz')
    X = train_data['X_train0']
    T = train_data['X_train1']
    y = train_data['y_train']


    m = 240
    epochs = 20000
    dim_x = 1
    lr = 0.001
    device = jax.devices('gpu')[0]

    branch = FCNN(m, [100], 100, activation='gelu')
    trunk = FCNN(dim_x, [100], 100, activation='gelu')

    model = DeepOnet(trunk, branch)


    optimizer = jax_opt.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, device)