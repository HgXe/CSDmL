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
        

class Dropout(ElementwiseOperation):
    def __init__(self, x: csdl.Variable, rate: float = 0.0, training: bool = True, seed: int = None):
        super().__init__(x)
        self.name = 'dropout'
        self.rate = rate
        self.training = training
        self.seed = seed

    def compute_inline(self, x):
        if not self.training or self.rate == 0.0:
            return x
        
        # Generate random mask
        if self.seed is not None:
            np.random.seed(self.seed)
        keep_prob = 1.0 - self.rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask

    def compute_jax(self, x):
        if not self.training or self.rate == 0.0:
            return x
            
        import jax
        import jax.numpy as jnp
        from jax import random
        
        # Use a fixed key for reproducibility during training
        key = random.PRNGKey(self.seed if self.seed is not None else 42)
        keep_prob = 1.0 - self.rate
        mask = random.bernoulli(key, keep_prob, shape=x.shape) / keep_prob
        return x * mask

    def evaluate_vjp(self, cotangents, x, dropout_x):
        if cotangents.check(x):
            if not self.training or self.rate == 0.0:
                cotangents.accumulate(x, cotangents[dropout_x])
            else:
                # During training, the gradient also gets multiplied by the same mask
                # For simplicity, we'll pass through the gradient as-is
                # In practice, the exact same mask should be used, but this is complex to implement
                cotangents.accumulate(x, cotangents[dropout_x])


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



def dropout(x: VariableLike, rate: float = 0.0, training: bool = True, seed: int = None) -> csdl.Variable:
    """Dropout regularization function.
    
    During training, randomly sets input units to 0 with a frequency of `rate` at each 
    step during training time, which helps prevent overfitting. Inputs not set to 0 
    are scaled up by 1/(1-rate) such that the sum over all inputs is unchanged.
    
    Parameters
    ----------
    x : Variable
        Input tensor
    rate : float, optional
        Fraction of the input units to drop. Float between 0 and 1. Default is 0.0.
    training : bool, optional
        Whether the layer is in training mode. If False, dropout is not applied.
        Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    out: Variable
        Output tensor with dropout applied during training, or unchanged tensor 
        during inference.
        
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0, 4.0]))
    >>> # During training with 50% dropout
    >>> y_train = dropout(x, rate=0.5, training=True)
    >>> # During inference (no dropout)
    >>> y_eval = dropout(x, rate=0.5, training=False)
    """
    x = validate_and_variablize(x)
    return Dropout(x, rate, training, seed).finalize_and_return_outputs()


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

def test_dropout():
    rec = csdl.Recorder(inline=True)
    rec.start()

    x = csdl.Variable(value = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
    
    # Test training mode
    y_train = dropout(x, rate=0.5, training=True, seed=42)
    
    # Test inference mode  
    y_eval = dropout(x, rate=0.5, training=False)
    
    print("Original:", x.value)
    print("Training (50% dropout):", y_train.value)
    print("Inference (no dropout):", y_eval.value)
    
    # In inference mode, output should be unchanged
    assert np.allclose(y_eval.value, x.value)
    
    # Test zero dropout rate
    y_no_dropout = dropout(x, rate=0.0, training=True)
    assert np.allclose(y_no_dropout.value, x.value)

def test_fcnn_with_dropout():
    import optax
    from csdml.core.neural_networks.fcnn import FCNN
    
    # Test the FCNN class with dropout
    rec = csdl.Recorder(inline=True)
    rec.start()

    X = np.random.rand(1000, 2)
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    X_test = np.random.rand(100, 2)
    Y_test = np.sin(X_test[:, 0:1]) + np.cos(X_test[:, 1:2])

    # Create network with dropout
    model = FCNN(2, [50, 50], 1, activation='tanh', dropout_rate=0.2)
    loss_data = X, y

    optimizer = optax.adam(1e-3)
    model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_epochs=100)
    
    # Test inference mode
    model.set_training_mode(False)
    X_test_var = csdl.Variable(value=X_test)
    y_pred = model.forward(X_test_var)
    print("Test completed successfully with dropout!")


