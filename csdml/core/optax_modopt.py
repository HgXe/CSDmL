from modopt import Optimizer
import optax


class OptaxOptimizer(Optimizer):
    def initialize(self):
        self.solver_name = 'Optax'
        self.options.declare('optimizer', types=optax.GradientTransformation)
        self.options.declare('maxiter', types=int, default=1000)

    def solve(self):
        optimizer = self.options['optimizer']
        maxiter = self.options['maxiter']
        