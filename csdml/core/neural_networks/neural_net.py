import csdl_alpha as csdl
import numpy as np
from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
import jax.numpy as jnp
from time import time
from typing import Union
import warnings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from optax import GradientTransformation


try:
    import optax
except ImportError:
    warnings.warn('optax not installed. Please install optax to use the train_jax_opt method')

try:
    from jax import jit as jjit
except ImportError:
    warnings.warn('jax not installed. Please install jax to use training methods')

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
    
    def _forward(self, x):
        return self.forward(x)

    def set_design_variables(self):
        for parameter in self.parameters:
            parameter.set_as_design_variable()
    
    def set_param_values(self, values:list[np.ndarray]):
        for parameter, value in zip(self.parameters, values):
            parameter.value = value
    
    def __call__(self, x):
        return self.forward(x)

    def _train(self, X, y):
        rec_outer = csdl.get_current_recorder()
        rec_outer.stop()

        # create a new recorder that only records the neural network
        rec_inner = csdl.Recorder()
        rec_inner.start()

        self.set_design_variables()

        # run the training loop
        y_pred = self._forward(X)

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
        
    def train_jax_opt(self, optimizer:Union[list, "GradientTransformation"], loss_data, num_batches=10, num_epochs=100, test_data=None, plot=True, device=None):
        """
        Train the neural network using JAX optimizers or optax optimizers

        Parameters
        ----------
        optimizer : Union[list, optax.GradientTransformation]
            JAX optimizer or optax optimizer
        loss_data : tuple
            loss_data = X_train, targets
        num_batches : int, optional
            Number of batches to use for training. The default is 10.
        num_epochs : int, optional
            Number of epochs to train. The default is 100.
        test_data : tuple, optional
            test_data = X_test, Y_test. The default is None.
        plot : bool, optional
            Whether to plot the loss history. The default is True.
        device : str, optional
            Device to use for training. The default is None.

        Returns
        -------
        loss_history : list
            List of training losses
        test_loss_history : list
            List of test losses
        best_param_vals : list
            List of best parameter values
        """
        
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
        y_pred = self._forward(X_var_batch)

        # compute the loss
        if self.loss_function == 'mse':
            loss = csdl.norm((Y_var_batch - y_pred))/batch_size
        elif callable(self.loss_function):
            loss = self.loss_function(self, X_var_batch, Y_var_batch, y_pred)
        else:
            raise ValueError('Invalid loss function')
        loss.set_as_objective()

        dvs = [var for var in rec_inner.design_variables.keys()]


        # build test function
        if test_data is not None:
            X_test, Y_test = test_data
            X_test_var = csdl.Variable(value=X_test)
            Y_test_var = csdl.Variable(value=Y_test)
            y_pred = self._forward(X_test_var)
            if self.loss_function == 'mse':
                test_loss = csdl.norm((Y_test_var - y_pred))/X_test.shape[0]
            elif callable(self.loss_function):
                test_loss = self.loss_function(self, X_test_var, Y_test_var, y_pred)
            jax_test_fn = jjit(create_jax_function(rec_inner.active_graph, outputs=[test_loss], inputs=dvs), device=device)

        # Build optimization step
        net_params = [jnp.array(dv.value) for dv in dvs]
        
        if isinstance(optimizer, optax.GradientTransformation):
            train_step = generate_optax_step(X_var_batch, Y_var_batch, optimizer)
            opt_state = optimizer.init(net_params)
        else:
            opt_init, opt_update, get_params = optimizer
            train_step = generate_jax_opt_step(X_var_batch, Y_var_batch, opt_update, get_params)
            opt_state = opt_init(net_params)
        train_step = jjit(train_step, device=device)

        # run optimization loop
        loss_history = []
        test_loss_history = []
        best_test_loss = np.inf
        best_params = net_params
        start = time()
        print_interval = max(1, num_epochs // 10)
        for epoch in range(num_epochs):
            # if epoch % print_interval == 0:
            print_status(epoch, num_epochs, loss_history, test_loss_history, start)

            for ibatch in range(num_batches):
                X_batch = X[ibatch*batch_size:(ibatch+1)*batch_size]
                Y_batch = Y[ibatch*batch_size:(ibatch+1)*batch_size]
                loss_data = X_batch, Y_batch
                loss, net_params, opt_state = train_step(ibatch, net_params, opt_state, loss_data)
                loss_history.append(float(loss[0]))
                if test_data is not None:
                    test_loss = jax_test_fn(*net_params)[0]
                    test_loss_history.append(float(test_loss[0]))
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_params = net_params

                if epoch == 0 and ibatch == 0:
                    # remove jitting time
                    start = time()

        end = time()
        msg = "training time for {0} epochs with {1} batches = {2:.1f} seconds"
        print()
        print(msg.format(num_epochs, num_batches, end-start))

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
        param_vals = [np.array(x) for x in net_params]

        # switch back to the outer recorder
        rec_inner.stop()
        rec_outer.start()

        # create new parameters and set the values
        self.init_parameters()
        self.set_param_values(param_vals)

        if test_data is not None:
            best_param_vals = [np.array(x) for x in best_params]
            return loss_history, test_loss_history, best_param_vals
        return loss_history, test_loss_history

def print_status(epoch, num_epochs, loss_history, test_loss_history, start):
    current_time = time()
    elapsed_time = current_time - start
    remaining_time = (elapsed_time / epoch * (num_epochs - epoch)) if epoch > 0 else 0
    training_loss = loss_history[-1] if loss_history else "N/A"
    test_loss = test_loss_history[-1] if test_loss_history else "N/A"

    # Construct the status message
    status_message = (
        f"Epoch {epoch}/{num_epochs} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Remaining: {remaining_time:.1f}s | "
        f"Train Loss: {training_loss} | "
        f"Test Loss: {test_loss}"
    )

    if epoch == 0:
        # Print normally for the first call
        print(status_message)
    else:
        # Overwrite the previous line
        print(f"\r{status_message}\033[K", end="")


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

    def train_step(step_i, net_params, opt_state, loss_data):
    
        outputs = jax_fn(*loss_data, *net_params)
        loss = outputs[0]
        grads = [out.reshape(param.shape) for out, param in zip(outputs[1:], net_params)]

        opt_state = opt_update(step_i, grads, opt_state)
        net_params = get_params(opt_state)

        return loss, net_params, opt_state

    return train_step

def generate_optax_step(X_var, Y_var, optimizer:"GradientTransformation"):
    '''
    
    Parameters
    ----------
    rec : csdl.Recorder
        DESCRIPTION.
    '''

    rec = csdl.get_current_recorder()

    dvs = [var for var in rec.design_variables.keys()]
    obj = [var for var in rec.objectives.keys()][0]
    grad = csdl.derivative(obj, dvs, as_block=False)
    grads = [grad[dv] for dv in dvs]

    jax_fn = create_jax_function(rec.active_graph, outputs=[obj]+grads, inputs=[X_var, Y_var] + dvs)

    def train_step(step_i, net_params, opt_state, loss_data):

        outputs = jax_fn(*loss_data, *net_params)
        loss = outputs[0]
        grads = [out.reshape(param.shape) for out, param in zip(outputs[1:], net_params)]

        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)

        return loss, net_params, opt_state

    return train_step