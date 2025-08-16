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
            interface = csdl.jax.create_jax_interface(inputs=list(self.inputs.keys()), outputs=self.outputs, graph=self.recorder.active_graph, device=device)
        elif backend == 'inline':
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
                 resume_state_file:str=None, save_state_file:str=None):
        """
        Generate samples using LatinHypercube. Supports batching/resuming.

        - n_samples: total number of samples you intend to produce overall (used to compute defaults).
        - batch_size: number of samples to produce in this call. If None, produce all remaining.
        - start: index to start sampling from (0-based). Ignored if resume_state_file provided.
        - seed: RNG seed for reproducible sequences.
        - resume_state_file: if provided and exists, loads {'seed', 'next_index'} and overrides start/seed.
        - save_state_file: if provided, saves {'seed', 'next_index'} after this batch so you can resume later.
        """
        import os
        import numpy as _np

        # resume state if requested
        if resume_state_file is not None and os.path.exists(resume_state_file):
            st = _np.load(resume_state_file)
            seed = int(st['seed'])
            start = int(st['next_index'])

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

            result = function(in_dict)
            if time_samples:
                print(f'Generated sample {n} in {time.time() - start_t} seconds')

            self._export_h5py(filename, {**in_dict, **result}, f'sample_{n}')

        # save resume state if requested (next index to sample)
        if save_state_file is not None:
            next_index = start + batch_size
            _np.savez(save_state_file, seed=int(seed), next_index=int(next_index))

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

    