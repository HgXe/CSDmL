import csdl_alpha as csdl
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
import h5py
import numpy as np

class Generator():
    def __init__(self, recorder:csdl.Recorder):
        self.recorder = recorder

        self.inputs = {}
        self.outputs = []
        self.loss = None

    def add_input(self, input, upper=None, lower=None):
        if not isinstance(input, csdl.Variable):
            raise ValueError('Input must be a csdl.Variable')
        if input in self.recorder.design_variables:
            if upper is None:
                upper = self.recorder.design_variables[input][2]
            if lower is None:
                lower = self.recorder.design_variables[input][1]
        bounds = [upper, lower]
        for i in range(2):
            bound = bounds[i]
            if bound is None:
                pass
            if isinstance(bound, (int, float)):
                bound = np.ones(input.shape) * bound
            if not isinstance(bound, np.ndarray):
                raise ValueError('Upper and lower bounds must be a scalar or a numpy array')
            if bound.shape != input.shape:
                raise ValueError('Upper and lower bounds must have the same shape as the input')
            bounds[i] = bound

        self.inputs[input] = bounds

    def add_output(self, output):
        # this method might not be needed if we can determine the output from the loss function
        self.outputs.append(output)

    def set_loss(self, loss):
        self.loss = loss

    def _estimate_input_probability_distribution(self):
        # Here we can use the GaussianProcessRegressor to estimate the probability distribution of the input variables
        # We compute the inputs from the design variables
        # we can either use LHS to sample the design variables or use values from the optimization
        pass


    def _build_generator_function(self, backend='jax', device='cpu'):
        if backend == 'jax':
            from csdl_alpha.src.operations.operation_subclasses import RandomOperation

            has_random_operations = any(
                isinstance(node, RandomOperation)
                for node in self.recorder.active_graph.node_table
            )
            if has_random_operations:
                # CSDL's standard interface currently drops the PRNG key during
                # its optional cost-analysis trace. Build the same jitted graph
                # directly for generators that contain random operations.
                import jax
                import numpy as np
                from csdl_alpha.backends.jax.graph_to_jax import create_jax_function

                input_variables = list(self.inputs.keys())
                output_variables = list(self.outputs)
                jax.config.update("jax_enable_x64", True)
                if device == 'gpu':
                    try:
                        jax_device = jax.devices('gpu')[0]
                    except Exception as error:
                        print(f"GPU not found: '{error}', falling back to CPU")
                        jax_device = jax.devices('cpu')[0]
                elif device == 'cpu':
                    jax_device = jax.devices('cpu')[0]
                else:
                    raise ValueError(f'Invalid device {device}')

                jax_function = create_jax_function(
                    self.recorder.active_graph,
                    output_variables,
                    input_variables,
                )
                jax_function = jax.jit(jax_function, device=jax_device)

                def interface(input_dict, prng_key=None):
                    values = [jax.numpy.asarray(input_dict[var]) for var in input_variables]
                    outputs = jax_function(*values, prng_key=prng_key)
                    return {
                        variable: np.asarray(value)
                        for variable, value in zip(output_variables, outputs)
                    }
            else:
                interface = csdl.jax.create_jax_interface(inputs=list(self.inputs.keys()), outputs=self.outputs, graph=self.recorder.active_graph, device=device)
        elif backend == 'inline':
            from csdl_alpha.src.operations.operation_subclasses import RandomOperation

            # A source-to-target extraction treats a zero-input random operation
            # as a hanging constant and therefore freezes its first inline value.
            # Execute the active graph when randomness is present so those
            # operations are evaluated afresh for every sample.
            if any(
                isinstance(node, RandomOperation)
                for node in self.recorder.active_graph.node_table
            ):
                generator_graph = self.recorder.active_graph
            else:
                generator_graph, _, _ = self.recorder.active_graph.extract_subgraph(self.inputs.keys(), self.outputs)
            def interface(input_dict):
                for key, value in input_dict.items():
                    key.value = value
                generator_graph.execute_inline()
                return {output: output.value for output in self.outputs}
        else:
            raise ValueError('Invalid backend')
        
        return interface

    def generate(self, filename:str='data', samples_per_dim:int=10, n_samples:int=None,
                 time_samples:bool=False, backend='jax', device='cpu',
                 batch_size:int=None, start:int=0, seed:int=0,
                 resume_state_file:str=None, save_state_file:str=None,
                 random_seed:int=None, save_inputs:bool=True):
        """
        Generate samples using LatinHypercube. Supports batching/resuming.

        - n_samples: total number of samples you intend to produce overall (used to compute defaults).
        - batch_size: number of samples to produce in this call. If None, produce all remaining.
        - start: index to start sampling from (0-based). Ignored if resume_state_file provided.
        - seed: RNG seed for reproducible sequences.
        - random_seed: seed for graph random operations. Each global sample index
          receives its own reproducible key. Defaults to ``seed``.
        - save_inputs: whether to include sampled inputs in each HDF5 group.
        - resume_state_file: if provided and exists, restores the sampling seed,
          random seed, and next sample index.
        - save_state_file: if provided, saves those values after this batch so
          generation can resume reproducibly.
        """
        import os
        import numpy as _np

        use_saved_random_seed = random_seed is None

        # resume state if requested
        if resume_state_file is not None and os.path.exists(resume_state_file):
            st = _np.load(resume_state_file)
            seed = int(st['seed'])
            start = int(st['next_index'])
            if use_saved_random_seed and 'random_seed' in st:
                random_seed = int(st['random_seed'])

        if random_seed is None:
            random_seed = seed

        function = self._build_generator_function(backend=backend, device=device)
        # in future, use the estimated input probability distribution to sample the input variables
        # for now we will just sample the inputs via LHS
        dims = []
        for input, bounds in self.inputs.items():
            dims.append(int(_np.prod(input.shape)))
            # apply default bounds
            if bounds[0] is None:
                bounds[0] = _np.ones(input.shape) * 1
            if bounds[1] is None:
                bounds[1] = _np.ones(input.shape) * 0

        total_dim = sum(dims)

        upper = _np.hstack([self.inputs[input][0].flatten() for input in self.inputs]).flatten()
        lower = _np.hstack([self.inputs[input][1].flatten() for input in self.inputs]).flatten()

        if n_samples is None:
            n_samples = samples_per_dim ** total_dim

        # determine how many to draw in this call
        if batch_size is None:
            batch_size = n_samples - start
        else:
            batch_size = min(batch_size, max(n_samples - start, 0))

        if batch_size <= 0:
            print('No samples to generate (start >= n_samples).')
            return

        # build sampler with reproducible RNG
        rng = _np.random.default_rng(seed)
        sampler = qmc.LatinHypercube(d=total_dim, seed=rng)

        # advance to start index
        if start > 0:
            sampler.fast_forward(start)

        # draw batch
        samples = sampler.random(batch_size)

        # scale samples into bounds
        scaler = upper - lower
        offset = lower
        samples = samples * scaler + offset

        print_interval = max(batch_size // 100, 1)
        import time

        for local_n, sample in enumerate(samples):
            n = start + local_n
            if local_n % print_interval == 0:
                print(f'Generating samples {n}-{min(n+print_interval, start+batch_size)} of {n_samples}')

            ind = 0
            in_dict = {}
            for i, input in enumerate(self.inputs):
                in_dict[input] = sample[ind:ind+dims[i]].reshape(input.shape)
                ind += dims[i]

            if time_samples:
                start_t = time.time()

            if backend == 'jax':
                from jax import random as _jax_random
                sample_key = _jax_random.fold_in(
                    _jax_random.PRNGKey(random_seed), n
                )
                result = function(in_dict, prng_key=sample_key)
            else:
                # RandomOperation.compute_inline uses NumPy's global RNG. Seed it
                # from the global sample index so resumed batches reproduce the
                # same random graph values as uninterrupted generation.
                sample_seed = _np.random.SeedSequence(
                    [int(random_seed), int(n)]
                ).generate_state(1)[0]
                _np.random.seed(sample_seed)
                result = function(in_dict)
            if time_samples:
                print(f'Generated sample {n} in {time.time() - start_t} seconds')

            saved_data = {**in_dict, **result} if save_inputs else result
            self._export_h5py(filename, saved_data, f'sample_{n}')

        # save resume state if requested (next index to sample)
        if save_state_file is not None:
            next_index = start + batch_size
            _np.savez(
                save_state_file,
                seed=int(seed),
                random_seed=int(random_seed),
                next_index=int(next_index),
            )

    def _export_h5py(self, filename:str, data:dict, groupname:str):
        """Save variables from the current recorder's node graph to an HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file to save the variables to.
        """
        import h5py
        
        if not filename.endswith('.hdf5'):
            filename = f'{filename}.hdf5'
        f = h5py.File(filename, 'a')

        inline_grp = f.create_group(groupname)
        name_counter_dict = {}
        for var, val in data.items():
            savename = self._get_savename(var, name_counter_dict)
            dset = inline_grp.create_dataset(savename, data=val)
            # The shape is already stored in the value
            dset.attrs['index'] = self.recorder.active_graph.node_table[var]
            if var.tags:
                dset.attrs['tags'] = var.tags
            if var.hierarchy is not None:
                dset.attrs['hierarchy'] = var.hierarchy
            if var.names:
                dset.attrs['names'] = var.names
        f.close()

    def _get_savename(self, key, name_counter_dict):
        if not key.names:
            if not key.namespace.prepend in name_counter_dict:
                name_counter_dict[key.namespace.prepend] = 0
            name_count = name_counter_dict[key.namespace.prepend]
            name_counter_dict[key.namespace.prepend] += 1
            if key.namespace.prepend is None:
                savename = f'variable_{name_count}'
            else:
                savename = f'{key.namespace.prepend}.variable_{name_count}'
        else:
            savename = key.names[0]
        return savename

    
