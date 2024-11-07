import csdl_alpha as csdl
from sklearn.gaussian_process import GaussianProcessRegressor

class Generator():
    def __init__(self, recorder:csdl.Recorder):
        self.recorder = recorder

        self.inputs = {}
        self.outputs = []
        self.loss = None

    def add_input(self, input, upper=None, lower=None):
        if input in self.recorder.design_variables:
            if upper is None:
                upper = self.recorder.design_variables[input][2]
            if lower is None:
                lower = self.recorder.design_variables[input][1]
        self.inputs[input] = (upper, lower)

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




    def _build_generator_function(self, backend='jax'):
        generator_graph, _, _ = self.recorder.active_graph.extract_subgraph(self.inputs.keys(), self.outputs)
        if backend == 'jax':
            interface = csdl.jax.create_jax_interface(inputs=self.inputs, outputs=self.outputs, graph=generator_graph)
        elif backend == 'inline':
            def interface(input_dict):
                for key, value in input_dict.items():
                    key.value = value
                generator_graph.execute_inline()
                return {output: output.value for output in self.outputs}
        else:
            raise ValueError('Invalid backend')
        
        return interface

    def generate(self, n_samples=1):
        pass


    

    