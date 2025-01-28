from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
import csdl_alpha as csdl
from scipy.special import erf
import numpy as np
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
from csdl_alpha.utils.typing import VariableLike


class GELu(ElementwiseOperation):
    def __init__(self, x:csdl.Variable):
        super().__init__(x)
        self.name = 'gelu'

    def compute_inline(self, x):
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def compute_jax(self, x):
        from jax.nn import gelu
        return gelu(x)
    
    def evaluate_vjp(self, cotangents, x, y):
        raise NotImplementedError('GELu does not have a VJP implementation')

class ReLuApproximate(ComposedOperation):
    def __init__(self, x:csdl.Variable):
        super().__init__(x)
        self.name = 'relu_approximate'

    def evaluate_composed(self, x):
        return x/2*(1+csdl.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
        
def gelu(x:VariableLike, approximate:bool=True)->csdl.Variable:
    """Gaussian error linear unit (GELu) activation function.

    Parameters
    ----------
    x : Variable

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    >>> csdl.gelu(x).value
    array([1.        , 0.        , 3.        , 0.        ])
    """
    x = validate_and_variablize(x)
    if approximate:
        return ReLuApproximate(x).finalize_and_return_outputs()
    else:
        raise(NotImplementedError('Only approximate version of GELu is implemented'))
        return ReLu(x).finalize_and_return_outputs()
    

