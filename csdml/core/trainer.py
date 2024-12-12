import csdl_alpha as csdl
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import qmc
import h5py
import numpy as np

class Trainer():
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

    def add_data(self, data_file):
        # Load the data from the file
        with h5py.File(data_file, 'r') as f:
            for key in f.keys():
                if key in self.inputs:
                    self.inputs[key] = f[key][:]
                elif key in self.outputs:
                    self.outputs[key] = f[key][:]


import pytorch