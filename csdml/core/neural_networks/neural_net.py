import csdl_alpha as csdl
import numpy as np
from csdml.core.activation_functions import gelu
from jax.example_libraries import optimizers as jax_opt
from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface, create_jax_function
from jax import jit as jjit
import jax.numpy as jnp
from time import time
import jax
from typing import Union


class NeuralNetwork():
    '''
    Base class for neural networks
    '''
    def __init__(self, loss_function:Union[str, callable], parameters:list[csdl.Variable]):
        self.loss_function = loss_function
        self.parameters = parameters

    def init_parameters(self):
        raise NotImplementedError('init_parameters method must be implemented in subclass')

    def forward(self, x):
        raise NotImplementedError('forward method must be implemented in subclass')
    
    def set_design_variables(self):
        for parameter in self.parameters:
            parameter.set_as_design_variable()
    
    def set_param_values(self, values:list[np.ndarray]):
        for parameter, value in zip(self.parameters, values):
            parameter.value = value
    
    def __call__(self, x):
        return self.forward(x)

    def train(self, X, y):
        rec_outer = csdl.get_current_recorder()
        rec_outer.stop()

        # create a new recorder that only records the neural network
        rec_inner = csdl.Recorder()
        rec_inner.start()

        self.set_design_variables()

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

        # extract values of the design variables
        param_vals = [param.value for param in self.parameters]

        # switch back to the outer recorder
        rec_inner.stop()
        rec_outer.start()

        # create new parameters and set the values
        self.init_parameters()
        self.set_param_values(param_vals)
        
    def train_jax_opt(self, optimizer:list, loss_data, num_batches=10, num_epochs=100, test_data=None, plot=True):
        # turn off the outer recorder
        rec_outer = csdl.get_current_recorder()
        rec_outer.stop()

        # create a new recorder that only records the neural network
        rec_inner = csdl.Recorder()
        rec_inner.start()

        self.init_parameters()
        self.set_design_variables()

        # create csdl variables for the loss data
        X, Y = loss_data
        batch_size = X.shape[0] // num_batches

        X_var_batch = csdl.Variable(shape=X[:batch_size].shape, value=0)
        Y_var_batch = csdl.Variable(shape=Y[:batch_size].shape, value=0)

        # run the training loop
        y_pred = self.forward(X_var_batch)

        # compute the loss
        if self.loss_function == 'mse':
            loss = csdl.norm((Y_var_batch - y_pred))
        elif callable(self.loss_function):
            loss = self.loss_function(Y_var_batch, y_pred)
        else:
            raise ValueError('Invalid loss function')
        loss.set_as_objective()

        # optimize the design variables
        opt_init, opt_update, get_params = optimizer

        train_step = generate_jax_opt_step(X_var_batch, Y_var_batch, opt_update, get_params)

        dvs = [var for var in rec_inner.design_variables.keys()]
        x0 = [jnp.array(dv.value) for dv in dvs]
        opt_state = opt_init(x0)

        # build test function
        if test_data is not None:
            X_test, Y_test = test_data
            y_pred = self.forward(X_test)
            if self.loss_function == 'mse':
                test_loss = csdl.norm((Y_test - y_pred))
            elif callable(self.loss_function):
                test_loss = self.loss_function(Y_test, y_pred)
            jax_test_fn = jjit(create_jax_function(rec_inner.active_graph, outputs=[test_loss], inputs=dvs))

        loss_history = []
        test_loss_history = []
        start = time()
        for epoch in range(num_epochs):
            for ibatch in range(num_batches):
                X_batch = X[ibatch*batch_size:(ibatch+1)*batch_size]
                Y_batch = Y[ibatch*batch_size:(ibatch+1)*batch_size]

                loss_data = X_batch, Y_batch
                
                loss, opt_state = train_step(ibatch, opt_state, loss_data)
                loss_history.append(float(loss[0]))
                if test_data is not None:
                    x_i = get_params(opt_state)
                    test_loss = jax_test_fn(*x_i)[0]
                    test_loss_history.append(float(test_loss[0]))


        end = time()
        msg = "training time for {0} iterations = {1:.1f} seconds"
        print(msg.format(num_batches, end-start))

        if plot:
            # plot the loss history
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)
            __=ax.plot(np.log10(loss_history))
            if test_data is not None:
                __=ax.plot(np.log10(test_loss_history))
                ax.legend(['train', 'test'])
            xlabel = ax.set_xlabel(r'${\rm step\ number}$')
            ylabel = ax.set_ylabel(r'$\log_{10}{\rm loss}$')
            title = ax.set_title(r'${\rm training\ history}$')
            plt.show()
        
        # extract values of the design variables
        xf = get_params(opt_state)
        param_vals = [np.array(x) for x in xf]

        # switch back to the outer recorder
        rec_inner.stop()
        rec_outer.start()

        # create new parameters and set the values
        self.init_parameters()
        self.set_param_values(param_vals)

def generate_jax_opt_step(X_var, Y_var, opt_update, get_params):
    '''
    
    Parameters
    ----------
    rec : csdl.Recorder
        DESCRIPTION.
    loss_data : tuple
        loss_data = X_train, targets
    '''
    rec = csdl.get_current_recorder()

    dvs = [var for var in rec.design_variables.keys()]
    obj = [var for var in rec.objectives.keys()][0]
    grad = csdl.derivative(obj, dvs, as_block=False)
    grads = [grad[dv] for dv in dvs]

    jax_fn = create_jax_function(rec.active_graph, outputs=[obj]+grads, inputs=[X_var, Y_var] + dvs)

    @jjit
    def train_step(step_i, opt_state, loss_data):
        net_params = get_params(opt_state)

        outputs = jax_fn(*loss_data, *net_params)
        loss = outputs[0]
        grads = [out.reshape(param.shape) for out, param in zip(outputs[1:], net_params)]

        return loss, opt_update(step_i, grads, opt_state)

    return train_step

