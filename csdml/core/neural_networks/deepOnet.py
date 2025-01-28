import csdl_alpha as csdl
import numpy as np
from csdml.core.neural_networks.fcnn import FCNN

class DeepOnet():
    def __init__(self, trunk:FCNN, branch:FCNN, output_dim:int=1, loss_function='mse'):
        self.trunk = trunk
        self.branch = branch
        self.output_dim = output_dim
        super().__init__(loss_function, self.trunk.parameters + self.branch.parameters)

    def init_parameters(self):
        self.trunk.init_parameters()
        self.branch.init_parameters()
        self.parameters = self.trunk.parameters + self.branch.parameters

    def forward(self, x, t):
        z = self.branch(x)
        y = self.trunk(t)

        # TODO: make outoput dim do something
        if self.output_dim != 1:
            raise NotImplementedError('Output dim other than 1 is not implemented')

        # TODO: consider adding bias here
        return z @ y

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