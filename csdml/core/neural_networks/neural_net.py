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
        
    def _compute_loss(self, X: np.ndarray, Y: np.ndarray , compute_grad: bool, additional_loss_functions: list[callable] = None):
        if compute_grad:
            # per-sample Jacobians; avoids B^2-size batched Jacobian
            batch_size = X.shape[0]
            X_var_shape = X.shape[1:]
            x_dim = int(np.prod(X_var_shape))
            X_vars = [csdl.Variable(shape=(1, *X_var_shape), value=X[i:i+1]) for i in range(batch_size)]
            X_var_batch = csdl.concatenate(X_vars, axis=0)
            Y_var_batch = csdl.Variable(shape=Y.shape, value=Y)

            y_pred = self._forward(X_var_batch)
            y_dim = int(np.prod(y_pred.shape[1:]))

            # assemble [value, gradient...]
            y_out = csdl.Variable(shape=(batch_size, y_dim, 1 + x_dim), value=0)
            y_out = y_out.set(csdl.slice[:, :, 0], csdl.reshape(y_pred, shape=(batch_size, y_dim)))

            # mode = 'reverse' if x_dim >= y_dim else 'fwd'
            mode = 'reverse' # TODO: only rev is supported by csdl currently
            for i in range(batch_size):
                jac_i = csdl.derivative(y_pred[i], X_vars[i], mode=mode)  # (y_dim, x_dim)
                y_out = y_out.set(csdl.slice[i, :, 1:], jac_i)

            y_pred = y_out


        else:
            X_var_batch = csdl.Variable(shape=X.shape, value=X)
            Y_var_batch = csdl.Variable(shape=Y.shape, value=Y)

            # run the training loop
            y_pred = self._forward(X_var_batch)

        # compute the loss
        if self.loss_function == 'mse':
            loss = csdl.sum((Y_var_batch - y_pred)**2)/np.prod(Y_var_batch.shape)
        elif callable(self.loss_function):
            loss = self.loss_function(self, X_var_batch, Y_var_batch, y_pred)
        else:
            raise ValueError('Invalid loss function')

        if additional_loss_functions is not None:
            additional_losses = []
            for additional_loss_fn in additional_loss_functions:
                add_loss = additional_loss_fn(self, X_var_batch, Y_var_batch, y_pred)
                additional_losses.append(add_loss)
            return loss, X_var_batch, Y_var_batch, additional_losses
        return loss, X_var_batch, Y_var_batch


    def train_jax_opt(
        self, 
        optimizer: Union[list, "GradientTransformation"], 
        loss_data: tuple, 
        num_batches: int = 10, 
        num_epochs: int = 100, 
        test_data: tuple = None, 
        test_interval: int = 1,
        plot: bool = True, 
        log_plot: bool = True, 
        device: 'xla_client.Device' = None, 
        trial: 'optuna.Trial' = None, 
        patience: int = None, 
        ema: 'optax.Ema' = None, 
        permute: bool = True, 
        prng_seed: int = 42, 
        add_jitter: bool = False, 
        jitter_std: float = 0.01, 
        compute_grad: bool = False,
        test_loss_functions: list[callable] = None,
        ):
        """
        Train the neural network using JAX optimizers or optax optimizers.

        This method trains the neural network by computing gradients with respect to
        the network parameters and updating them using the specified optimizer. It
        supports batched training, early stopping, exponential moving average (EMA),
        and integration with Optuna for hyperparameter optimization.

        Parameters
        ----------
        optimizer : Union[list, optax.GradientTransformation]
            The optimizer to use for training. Can be either:
            - An optax.GradientTransformation object (e.g., optax.adam, optax.sgd)
            - A tuple of (opt_init, opt_update, get_params) for custom JAX optimizers
        loss_data : tuple
            Training data as (X_train, Y_train) where X_train and Y_train are numpy arrays.
        num_batches : int, optional
            Number of batches to divide the training data into. The default is 10.
        num_epochs : int, optional
            Number of training epochs to run. The default is 100.
        test_data : tuple, optional
            Validation/test data as (X_test, Y_test). If provided, test loss is
            computed at intervals and used for early stopping. The default is None.
        test_interval : int, optional
            Interval (in batches) at which to evaluate the test loss. The default is 1.
        plot : bool, optional
            Whether to save a plot of the loss history to 'loss_history.png'.
            The default is True.
        log_plot : bool, optional
            Whether to use log scale for the loss history plot y-axis. The default is True.
        device : xla_client.Device, optional
            JAX device to use for JIT compilation (e.g., 'gpu:0', 'cpu'). 
            The default is None (uses default device).
        trial : optuna.Trial, optional
            Optuna trial object for hyperparameter optimization. If provided,
            best test loss is reported to Optuna at each epoch. The default is None.
        patience : int, optional
            Number of test intervals to wait without improvement before early stopping.
            Only used if test_data is provided. The default is None (no early stopping).
        ema : optax.Ema, optional
            Optax exponential moving average wrapper for parameter averaging.
            If provided, an EMA version of parameters is maintained and used.
            The default is None.
        permute : bool, optional
            Whether to randomly permute the training data at each epoch. 
            The default is True.
        prng_seed : int, optional
            Random seed for JAX PRNG key initialization. The default is 42.
        add_jitter : bool, optional
            Whether to add Gaussian noise to input data during training for regularization.
            The default is False.
        jitter_std : float, optional
            Standard deviation of the Gaussian noise added to inputs if add_jitter is True.
            The default is 0.01.
        compute_grad : bool, optional
            Whether to compute and include per-sample Jacobians in the loss computation.
            This is useful for computing gradients with respect to inputs. 
            The default is False.
        test_loss_functions : list[callable], optional
            A list of additional loss functions to compute on the test set at each test interval.
            Each function should take arguments (self, X_test_var, Y_test_var, y_pred) and return a scalar loss. The computed losses will be logged but not used for early stopping.
            The default is None.

        Returns
        -------
        loss_history : list
            List of training loss values computed at each batch step.
        test_loss_history : list (optional)
            List of test loss values. Only returned if test_data is provided.
        best_param_vals : list
            List of best parameter values found during training. Parameters are
            ordered according to self.parameters.

        Raises
        ------
        optuna.TrialPruned
            If an Optuna trial is provided and should be pruned based on its criteria.

        Notes
        -----
        - The method temporarily stops the outer recorder and creates a new inner
          recorder to isolate the training graph.
        - After training completes, the network is set to inference mode via
          set_training_mode(training=False).
        - If EMA is used, the EMA-averaged parameters are returned as best_param_vals.
        - Early stopping (patience) compares against test loss improvements.
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

        X_batch = X[:batch_size]
        Y_batch = Y[:batch_size]
        
        loss, X_var_batch, Y_var_batch = self._compute_loss(X_batch, Y_batch, compute_grad)
        
        loss.set_as_objective()

        dvs = [var for var in rec_inner.design_variables.keys()]

        # build test function
        if test_data is not None:
            X_test, Y_test = test_data
            self.set_training_mode(training=False)
            if test_loss_functions is not None:
                test_loss, X_test_var, Y_test_var, additional_test_losses = self._compute_loss(X_test, Y_test, compute_grad, additional_loss_functions=test_loss_functions)
                jax_test_fn = jjit(create_jax_function(rec_inner.active_graph, outputs=[test_loss]+additional_test_losses, inputs=dvs), device=device)
            else:
                test_loss, X_test_var, Y_test_var = self._compute_loss(X_test, Y_test, compute_grad)
                jax_test_fn = jjit(create_jax_function(rec_inner.active_graph, outputs=[test_loss], inputs=dvs), device=device)
            self.set_training_mode(training=True)

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
        additional_loss_histories = [[] for _ in range(len(test_loss_functions))] if test_loss_functions is not None else None
        best_test_loss = np.inf
        best_loss = np.inf
        best_params = net_params
        start = time()
        print_interval = max(1, num_epochs // 10)
        np_rng = np.random.default_rng(seed=42)
        wait = 0
        early_stopped = False
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

                # add jitter to the input data if specified
                if add_jitter:
                    prng_key, subkey = jax.random.split(prng_key)
                    jitter = jax.random.normal(subkey, X_batch.shape) * jitter_std
                    X_batch += jitter

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
                    if (ibatch + num_batches*epoch) % test_interval == 0:
                        test_loss = jax_test_fn(*net_params, prng_key=subkey)
                        if test_loss_functions is not None:
                            additional_losses = test_loss[1:]
                            test_loss = test_loss[0]
                        
                        test_loss = float(test_loss[0])
                        test_loss_history.append(test_loss)
                        if test_loss_functions is not None:
                            for i, loss in enumerate(additional_losses):
                                additional_loss_histories[i].append(float(loss[0]))
                        if test_loss < best_test_loss:
                            best_test_loss = test_loss
                            best_params = net_params
                            decreased = True
                else:
                    loss_check = float(loss[0])
                    if loss_check < best_loss:
                        best_loss = loss_check
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
                        if patience is not None and wait > patience * test_interval:
                            print()
                            print(f'Early stopping at epoch {epoch}')
                            early_stopped = True
                            break
            
            # report the best test loss to optuna
            if trial is not None and test_data is not None:
                trial.report(best_test_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if early_stopped:
                break
                
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

            # Plot training loss
            __=plot_fn(loss_history, label='train')
            

            # Plot test loss at intervals
            if test_data is not None:
                test_steps = [i * test_interval for i in range(len(test_loss_history))]

                # Plot additional test losses if they exist at intervals
                if additional_loss_histories is not None:
                    for i, hist in enumerate(additional_loss_histories):
                        __=plot_fn(test_steps, hist, label=f'test_{i}')

                # plot main test loss (same as training loss)
                __=plot_fn(test_steps, test_loss_history, label='test')
                
                # Mark and annotate minimum test loss
                min_test_idx = np.argmin(test_loss_history)
                min_test_step = test_steps[min_test_idx]
                min_test_loss = test_loss_history[min_test_idx]
                ax.plot(min_test_step, min_test_loss, 'r*', markersize=15)
                ax.annotate(f'min: {min_test_loss:.4g}', 
                           xy=(min_test_step, min_test_loss),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
                ax.legend()
            
            # Mark epochs on the plot
            steps_per_epoch = num_batches
            for epoch in range(1, num_epochs + 1):
                epoch_step = epoch * steps_per_epoch
                if epoch_step < len(loss_history):
                    ax.axvline(epoch_step, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            
            xlabel = ax.set_xlabel(r'${\rm step\ number}$')
            ylabel = ax.set_ylabel(r'${\rm loss}$')
            title = ax.set_title(r'${\rm training\ history}$')
            plt.savefig('loss_history.png', dpi=300)
            plt.close()
        
        # extract values of the design variables
        param_vals = [np.array(x) for x in best_params]

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
            if test_loss_functions is not None:
                return loss_history, test_loss_history, additional_loss_histories, param_vals
            return loss_history, test_loss_history, param_vals
        return loss_history, param_vals

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