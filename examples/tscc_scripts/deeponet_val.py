import optax
import jax
import csdl_alpha as csdl
import csdml
import numpy as np


# test the DeepOnet class
rec = csdl.Recorder(inline=True)
rec.start()

test_data = np.load('test.npz')
X_test = test_data['X_test0']
T_test = test_data['X_test1']
Y_test = test_data['y_test']

train_data = np.load('train.npz')
X = train_data['X_train0']
T = train_data['X_train1']
y = train_data['y_train']

# test the DeepOnet class
rec = csdl.Recorder(inline=True)
rec.start()

m = 100
epochs = 50000
dim_x = 1
lr = 0.001
device = jax.devices('gpu')[0]
# device = jax.devices('cpu')[0]

branch = csdml.FCNN(m, [40], 40, activation='relu')
trunk = csdml.FCNN(dim_x, [40], 40, activation='relu')

model = csdml.DeepOnet(trunk, branch)
loss_data = X, T, y


optimizer = optax.adam(1e-3)
loss_history, test_loss_history, best_param_vals = model.train_jax_opt(optimizer, loss_data, test_data=(X_test, T_test, Y_test), num_batches=1, num_epochs=epochs, device=device)

# save loss history
loss_history = np.array(loss_history)
test_loss_history = np.array(test_loss_history)

np.savez('loss_history.npz', loss_history=loss_history, test_loss_history=test_loss_history)