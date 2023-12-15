import math
import numpy as np
layer_op = [4.8, 1.21, 2.385]

# E = 2.7182818
E = math.e

# exp_values = [E**o for o in layer_op]
# print(exp_values)

# norm_base = sum(exp_values)

# norm_values = [o/norm_base for o in exp_values]

# print(norm_values, sum(norm_values))

# # layer_op = np.array(layer_op)

# exp_values = np.exp(layer_op)
# print("np", exp_values)
  
# norm_values = exp_values / np.sum(exp_values)

# print("np", norm_values, sum(norm_values))

# lets process batch

layer_ops = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.3], [1.41, 1.051, 0.026]]
exp_values = np.exp(layer_ops)
print("batchnp\n", exp_values)
print(np.sum(layer_ops, axis=1, keepdims=True))
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values, '\n', np.sum(norm_values, axis=1, keepdims=True))

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
X, y = spiral_data(100, 3)
class Layer_Dense:
    def __init__(self, n_inputs, n_features) -> None:
        # weights are n_features x n_inputs, but to save a transpose operation we do n_inputs x n_features
        self.weights = 0.1 * np.random.randn(n_inputs, n_features)
        # print("weights", self.weights.shape)
        self.biases = np.zeros((1, n_features))
    
    def forward(self, inputs):
        # print("ip and weights", inputs.shape, self.weights.shape)
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
        
class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        
     
class Activation_Softmax:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        
           
layer1 = Layer_Dense(X.shape[1], 5)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
layer2 = Layer_Dense(5, 10)
layer3 = Layer_Dense(10, 5)
layer4 = Layer_Dense(5, 2)

layer1.forward(X)
# print(layer1.outputs)
activation1.forward(layer1.outputs)
print(activation1.outputs)
layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)
# print(activation1.outputs)
# print(activation2.outputs[:5])#, '\n', np.sum(activation2.outputs, axis=1, keepdims=True))
# layer2.forward(activation1.outputs)
# layer3.forward(layer2.outputs)
# layer4.forward(layer3.outputs)
# print(layer4.outputs)
