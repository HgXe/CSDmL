import csdl_alpha as csdl
import numpy as np
from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
import jax.numpy as jnp
import jax
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
    
    def get_param_values(self):
        return [parameter.value for parameter in self.parameters]
    
    def set_training_mode(self, training: bool = True):
        """
        Set the training mode for the neural network.
        
        This is a base implementation that does nothing by default.
        Subclasses can override this method to implement training-specific
        behavior (e.g., enabling/disabling dropout).
        
        Parameters
        ----------
        training : bool, optional
            Whether to set the network to training mode (True) or inference mode (False).
            Default is True.
        """
        pass

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
            n = np.prod(y.shape)
            loss = csdl.norm((y - y_pred))/n
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
        
    def train_jax_opt(self, optimizer:Union[list, "GradientTransformation"], loss_data, 
                      num_batches=10, num_epochs=100, test_data=None, plot=True, log_plot=True, device=None, 
                      trial=None, patience=None, ema=None, permute=True, prng_seed=42):
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
        if trial is not None:
            import optuna
        
        # Set training mode
        self.set_training_mode(training=True)
        
        # get current values of parameters
        current_param_vals = self.get_param_values()
        
        # turn off the outer recorder
        rec_outer = csdl.get_current_recorder()
        rec_outer.stop()

        # create a new recorder that only records the neural network
        rec_inner = csdl.Recorder()
        rec_inner.start()

        self.init_parameters()
        self.set_param_values(current_param_vals)
        self.set_design_variables()

        # create csdl variables for the loss data
        X, Y = loss_data
        X_device = jnp.array(X)
        Y_device = jnp.array(Y)
        batch_size = X.shape[0] // num_batches

        X_var_batch = csdl.Variable(shape=X[:batch_size].shape, value=0)
        Y_var_batch = csdl.Variable(shape=Y[:batch_size].shape, value=0)

        # run the training loop
        y_pred = self._forward(X_var_batch)

        # compute the loss
        if self.loss_function == 'mse':
            loss = csdl.sum((Y_var_batch - y_pred)**2)/np.prod(Y_var_batch.shape)
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
                test_loss = csdl.sum((Y_test_var - y_pred)**2)/np.prod(Y_test_var.shape)
            elif callable(self.loss_function):
                test_loss = self.loss_function(self, X_test_var, Y_test_var, y_pred)
            jax_test_fn = jjit(create_jax_function(rec_inner.active_graph, outputs=[test_loss], inputs=dvs), device=device)

        # Build optimization step
        net_params = [jnp.array(dv.value) for dv in dvs]
        
        if isinstance(optimizer, optax.GradientTransformation):
            train_step_base = generate_optax_step(X_var_batch, Y_var_batch, optimizer, debug=False)
            opt_state = optimizer.init(net_params)
        else:
            opt_init, opt_update, get_params = optimizer
            train_step_base = generate_jax_opt_step(X_var_batch, Y_var_batch, opt_update, get_params)
            opt_state = opt_init(net_params)
        
        if ema is not None:
            def train_step_ema(step_i, net_params, opt_state, ema_state, loss_data, prng_key=None):
                loss, net_params, opt_state = train_step_base(step_i, net_params, opt_state, loss_data, prng_key=prng_key)
                net_params, ema_state = ema.update(net_params, ema_state)
                return loss, net_params, opt_state, ema_state
                
            train_step = jjit(train_step_ema, device=device)
        else:
            train_step = jjit(train_step_base, device=device)

        # initialize ema
        if ema is not None:
            ema_state = ema.init(net_params)

        # initialize random number generator
        prng_key = jax.random.PRNGKey(prng_seed)

        # run optimization loop
        loss_history = []
        test_loss_history = []
        best_test_loss = np.inf
        best_params = net_params
        start = time()
        print_interval = max(1, num_epochs // 10)
        np_rng = np.random.default_rng(seed=42)
        wait = 0
        for epoch in range(num_epochs):
            perm = np_rng.permutation(X.shape[0])
            decreased = False

            for ibatch in range(num_batches):
                lo = ibatch * batch_size
                hi = (ibatch + 1) * batch_size
                if permute:
                    idx = perm[lo:hi]
                else:
                    idx = slice(lo, hi)
                X_batch = X_device[idx]
                Y_batch = Y_device[idx]

                # split prng key for each batch (eg for dropout)
                prng_key, subkey = jax.random.split(prng_key)

                # X_batch = X[ibatch*batch_size:(ibatch+1)*batch_size]
                # Y_batch = Y[ibatch*batch_size:(ibatch+1)*batch_size]
                loss_data = X_batch, Y_batch
                if ema is not None:
                    loss, net_params, opt_state, ema_state = train_step(epoch*num_batches + ibatch, net_params, opt_state, ema_state, loss_data, prng_key=subkey)
                else:
                    loss, net_params, opt_state = train_step(ibatch+num_batches*epoch, net_params, opt_state, loss_data, prng_key=subkey)

                loss_history.append(float(loss[0]))
                if test_data is not None:
                    test_loss = jax_test_fn(*net_params)[0]
                    test_loss = float(test_loss[0])
                    test_loss_history.append(test_loss)
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_params = net_params
                        decreased = True

                print_status(epoch, num_epochs, ibatch+1, num_batches, loss_history, test_loss_history, start)

                if epoch == 0 and ibatch == 0:
                    # remove jitting time
                    start = time()

            if test_data is not None:
                if decreased:
                    wait = 0
                else:
                    wait += 1
                    if patience is not None and wait > patience * num_batches:
                        print()
                        print(f'Early stopping at epoch {epoch}')
                        break
            
            # report the best test loss to optuna
            if trial is not None and test_data is not None:
                trial.report(best_test_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
        end = time()
        msg = "training time for {0} epochs with {1} batches = {2:.1f} seconds"
        print()
        print(msg.format(epoch+1, num_batches, end-start))

        if plot:
            # plot the loss history
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1)

            if log_plot:
                plot_fn = ax.semilogy
            else:
                plot_fn = ax.plot

            __=plot_fn(loss_history)
            if test_data is not None:
                __=plot_fn(test_loss_history)
                ax.legend(['train', 'test'])
            xlabel = ax.set_xlabel(r'${\rm step\ number}$')
            ylabel = ax.set_ylabel(r'${\rm loss}$')
            title = ax.set_title(r'${\rm training\ history}$')
            plt.savefig('loss_history.png', dpi=300)
            plt.close()
        
        # extract values of the design variables
        param_vals = [np.array(x) for x in net_params]

        for dv, val in zip(dvs, param_vals):
            dv.value = val

        param_vals = []
        for parameter in self.parameters:
            if isinstance(parameter, csdl.Variable):
                param_vals.append(parameter.value)
            else:
                param_vals.append(parameter)

        # switch back to the outer recorder
        rec_inner.stop()
        rec_outer.start()

        # create new parameters and set the values
        self.init_parameters()
        self.set_param_values(param_vals)
        
        # Restore inference mode after training
        self.set_training_mode(training=False)

        if test_data is not None:
            best_param_vals = [np.array(x) for x in best_params]
            return loss_history, test_loss_history, best_param_vals
        return loss_history, test_loss_history

def print_status(epoch, num_epochs, step, num_steps, loss_history, test_loss_history, start):
    """
    Print a one-line status message at each batch step.
    """
    current_time = time()
    elapsed_time = current_time - start
    total_steps_done = epoch * num_steps + step
    total_steps = num_epochs * num_steps
    remaining_time = (elapsed_time / total_steps_done * (total_steps - total_steps_done)) if total_steps_done > 0 else 0.0

    train_loss = loss_history[-1] if loss_history else "N/A"
    test_loss  = test_loss_history[-1] if test_loss_history else "N/A"

    # --- changed code starts here ---
    # format numeric losses with up to 4 significant digits (scientific if needed)
    if isinstance(train_loss, (int, float)):
        train_loss_str = f"{train_loss:.4g}"
    else:
        train_loss_str = str(train_loss)
    if isinstance(test_loss, (int, float)):
        test_loss_str = f"{test_loss:.4g}"
    else:
        test_loss_str = str(test_loss)

    # fixed width for step to match num_steps digits
    step_width = len(str(num_steps))
    # --- changed code ends here ---

    msg = (
        f"Epoch {epoch+1}/{num_epochs} "
        f"Step {step:>{step_width}}/{num_steps} | "
        f"Elapsed: {elapsed_time:.1f}s | "
        f"Remain: {remaining_time:.1f}s | "
        f"Train Loss: {train_loss_str} | "
        f"Test Loss: {test_loss_str}"
    )
    if total_steps_done == 1:
        # first print, no overwrite
        print(msg)
    else:
        # overwrite previous line
        print(f"\r{msg}\033[K", end="")


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

def generate_optax_step(X_var, Y_var, optimizer:"GradientTransformation", debug=False):
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

    def train_step(step_i, net_params, opt_state, loss_data, prng_key=None):

        outputs = jax_fn(*loss_data, *net_params, prng_key=prng_key)
        loss = outputs[0]
        grads = [out.reshape(param.shape) for out, param in zip(outputs[1:], net_params)]

        if debug:
            gnorm = optax.global_norm(grads)
            jax.debug.print("batch {b}: loss={l}, |g|={g}",
                            b=step_i, l=loss[0], g=gnorm)

        # value_fn = lambda x: jax_fn(*loss_data, *x)[0][0]

        # updates, opt_state = optimizer.update(grads, opt_state, net_params,
        #                                       value=loss[0], grad=grads, value_fn=value_fn)
        
        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)

        return loss, net_params, opt_state

    return train_step