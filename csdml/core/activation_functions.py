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
        
class Softplus(ComposedOperation):
    def __init__(self, x:csdl.Variable, beta:float=1.0):
        super().__init__(x)
        self.name = 'softplus'
        self.beta = beta

    def evaluate_composed(self, x):
        return csdl.log(1 + csdl.exp(self.beta*x))/self.beta

class ParametricReLU(ElementwiseOperation):
    def __init__(self, x:csdl.Variable, alpha:float=0.0):
        super().__init__(x)
        self.name = 'p_relu'
        self.alpha = alpha

    def compute_inline(self, x):
        return np.maximum(x, self.alpha*x)
    
    def compute_jax(self, x):
        import jax.numpy as jnp
        return jnp.maximum(x, self.alpha*x)
    
    def evaluate_vjp(self, cotangents, x, prelu_x):
        if cotangents.check(x):
            cotangents.accumulate(x, d_parametric_relu(x, self.alpha)*cotangents[prelu_x])

class ParametricReLUDerivative(ElementwiseOperation):
    def __init__(self, x:csdl.Variable, alpha:float=0.0):
        super().__init__(x)
        self.name = 'p_relu_derivative'
        self.alpha = alpha

    def compute_inline(self, x):
        return np.heaviside(x, 0.5) + self.alpha*np.heaviside(-x, 0.5)
    
    def compute_jax(self, x):
        import jax.numpy as jnp
        return jnp.heaviside(x, 0.5) + self.alpha*jnp.heaviside(-x, 0.5)
    
    def evaluate_vjp(self, cotangents, x, dprelu_u):
        if cotangents.check(x):
            cotangents.accumulate(x, 0*cotangents[dprelu_u])
        


def softplus(x:VariableLike, beta:float=1.0)->csdl.Variable:
    """Softplus activation function.

    Parameters
    ----------
    x : Variable
    beta : float, optional
        by default 1.0

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    >>> csdl.softplus(x).value
    array([1.31326169, 0.12692801, 3.04858735, 0.01814993])
    """
    x = validate_and_variablize(x)
    return Softplus(x, beta).finalize_and_return_outputs()

def relu_approximate(x:VariableLike)->csdl.Variable:
    """Approximate ReLu activation function.

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
    >>> csdl.relu_approximate(x).value
    array([0.5       , 0.        , 3.        , 0.        ])
    """
    x = validate_and_variablize(x)
    return ReLuApproximate(x).finalize_and_return_outputs()

def parametric_relu(x:VariableLike, alpha:float=0.0)->csdl.Variable:
    """Parametric ReLu activation function.

    Parameters
    ----------
    x : Variable
    alpha : float, optional
        by default 0.0

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    >>> csdl.parametric_relu(x, alpha=0.5).value
    array([1. , -1. , 3. , -2. ])
    """
    x = validate_and_variablize(x)
    return ParametricReLU(x, alpha).finalize_and_return_outputs()

def d_parametric_relu(x:VariableLike, alpha:float=0.0)->csdl.Variable:
    """Derivative of Parametric ReLu activation function.

    Parameters
    ----------
    x : Variable
    alpha : float, optional
        by default 0.0

    Returns
    -------
    out: Variable

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    >>> csdl.d_parametric_relu(x, alpha=0.5).value
    array([1. , 0.5, 1. , 0.5])
    """
    x = validate_and_variablize(x)
    return ParametricReLUDerivative(x, alpha).finalize_and_return_outputs()



def test_softplus():
    rec = csdl.Recorder(inline=True)
    rec.start()

    x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    y = softplus(x)
    dy = csdl.derivative(y, x, elementwise=True)

    assert np.allclose(y.value, np.log(1 + np.exp(x.value)))
    assert np.allclose(np.diag(dy.value), 1/(1 + np.exp(-x.value)))

    # plot softplus and its derivative
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(-5, 5, 100)
    # x_var = csdl.Variable(value = x)
    # y = softplus(x).value
    # dy = np.diag(csdl.derivative(softplus(x_var), x_var, elementwise=True).value)
    # __=ax.plot(x, y)
    # __=ax.plot(x, dy)
    # ax.legend(['Softplus', 'Derivative'])
    # plt.show()

def test_relu_approximate():
    rec = csdl.Recorder(inline=True)
    rec.start()

    x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    y = relu_approximate(x)
    dy = csdl.derivative(y, x, elementwise=True)

    assert np.allclose(y.value, 0.5 * x.value * (1 + np.tanh(np.sqrt(2/np.pi)*(x.value+0.044715*x.value**3))))
    # assert np.allclose(np.diag(dy.value), 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x.value+0.044715*x.value**3))) + 0.5*x.value*(1 - np.tanh(np.sqrt(2/np.pi)*(x.value+0.044715*x.value**3))**2*(np.sqrt(2/np.pi)*(1+3*0.044715*x.value**2))))

    # plot relu_approximate and its derivative
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(-5, 5, 100)
    # x_var = csdl.Variable(value = x)
    # y = relu_approximate(x_var).value
    # dy = np.diag(csdl.derivative(relu_approximate(x_var), x_var, elementwise=True).value)
    # __=ax.plot(x, y)
    # __=ax.plot(x, dy)
    # ax.legend(['ReLU Approximate', 'Derivative'])
    # plt.show()

def test_parametric_relu():
    rec = csdl.Recorder(inline=True)
    rec.start()

    x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
    y = parametric_relu(x, alpha=0.5)
    dy = csdl.derivative(y, x, elementwise=True)

    assert np.allclose(y.value, np.maximum(x.value, 0.5*x.value))
    assert np.allclose(np.diag(dy.value), np.heaviside(x.value, 0.5) + 0.5*np.heaviside(-x.value, 0.5))

    # plot parametric_relu and its derivative
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # x = np.linspace(-5, 5, 100)
    # x_var = csdl.Variable(value = x)
    # y = parametric_relu(x_var, alpha=0.5).value
    # dy = np.diag(csdl.derivative(parametric_relu(x_var, alpha=0.5), x_var, elementwise=True).value)
    # __=ax.plot(x, y)
    # __=ax.plot(x, dy)
    # ax.legend(['Parametric ReLU', 'Derivative'])
    # plt.show()

if __name__ == '__main__':
    test_softplus()
    test_relu_approximate()
    test_parametric_relu()




# def gelu(x:VariableLike, approximate:bool=True)->csdl.Variable:
#     """Gaussian error linear unit (GELu) activation function.

#     Parameters
#     ----------
#     x : Variable

#     Returns
#     -------
#     out: Variable

#     Examples
#     --------
#     >>> recorder = csdl.Recorder(inline = True)
#     >>> recorder.start()
#     >>> x = csdl.Variable(value = np.array([1.0, -2.0, 3.0, -4.0]))
#     >>> csdl.gelu(x).value
#     array([1.        , 0.        , 3.        , 0.        ])
#     """
#     x = validate_and_variablize(x)
#     if approximate:
#         return ReLuApproximate(x).finalize_and_return_outputs()
#     else:
#         raise(NotImplementedError('Only approximate version of GELu is implemented'))
#         return ReLu(x).finalize_and_return_outputs()
    

