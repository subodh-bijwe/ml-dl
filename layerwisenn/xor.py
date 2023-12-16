from dense_layer import Dense
from tanh import Tanh
from mse import mse, mse_prime
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
print("checking shapes:", X.shape, Y.shape)

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

epochs = 10
learning_rate = 0.1
verbose = True
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = predict(network, x)
        error += mse(y, output)
        
        grad = mse_prime(y, output)
        
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            
    error /= len(x)
    
    if verbose:
        print(f"Epoch: {e + 1}/{epochs}, Error={error:.5f}")