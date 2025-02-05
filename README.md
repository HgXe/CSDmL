# CSDmL

<!---
[![Python](https://img.shields.io/pypi/pyversions/lsdo_project_template)](https://img.shields.io/pypi/pyversions/lsdo_project_template)
[![Pypi](https://img.shields.io/pypi/v/lsdo_project_template)](https://pypi.org/project/lsdo_project_template/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

[![GitHub Actions Test Badge](https://github.com/HgXe/CSDmL/actions/workflows/actions.yml/badge.svg)](https://github.com/HgXe/CSDmL/actions)
[![Forks](https://img.shields.io/github/forks/HgXe/CSDmL.svg)](https://github.com/HgXe/CSDmL/network)
[![Issues](https://img.shields.io/github/issues/HgXe/CSDmL.svg)](https://github.com/HgXe/CSDmL/issues)


Basic machine learning library for CSDL. WARNING: WIP - This library is still under development and may not be stable.


# Installation

## Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line
```sh
pip install git+https://github.com/HgXe/CSDmL.git
```

Installing JAX and optax is highly recommended for the library to work properly. See [JAX](https://jax.readthedocs.io/en/latest/installation.html#installation), [Optax](https://optax.readthedocs.io/en/latest/) for installation instructions.

Tested with Python 3.8 and 3.9. Other versions may work but are not guaranteed.

<!-- **Enabled by**: `packages=find_packages()` in the `setup.py` file. -->

## Installation instructions for developers
To install `csdml`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
git clone https://github.com/HgXe/CSDmL.git
pip install -e ./CSDmL
```

# Usage
To use the library, import the package and use the functions as needed.

```python
import csdml
import optax
import numpy as np
import csdl_alpha as csdl

# start csdl recorder
rec = csdl.Recorder(inline=True)
rec.start()

# generate training and test data
X = np.random.rand(10000, 1)*2*np.pi
y = np.sin(X)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)*2*np.pi
Y_test = np.sin(X_test)

# define neural network
activation = ['relu', 'tanh', 'tanh', 'tanh', 'tanh']
model = csdml.FCNN(1, [20, 20, 20, 20], 1, activation=activation)
loss_data = X, y

# train model
optimizer = optax.adam(1e-3)
model.train_jax_opt(optimizer, loss_data, test_data=(X_test, Y_test), num_epochs=1000)

# plot results
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

y_pred = model.forward(X_test).value
__=ax.plot(X_test, y_pred)
__=ax.plot(X_test, Y_test)

ax.legend(['Predicted', 'True'])

plt.show()

```

# For Developers
For details on documentation, refer to the README in `docs` directory.

For details on testing/pull requests, refer to the README in `tests` directory.

# License
This project is licensed under the terms of the **GNU Lesser General Public License v3.0**.
