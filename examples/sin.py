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