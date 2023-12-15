import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
np.random.seed(0)
X = np.array([
        [1, 2, 3, 2.5], 
        [2.0, 5.0, -1.0, 2.0], 
        [-1.5, 2.7, 3.3, -0.8]
    ])
X, y = spiral_data(100, 3)
X = np.array(X)
# print(X, y, X.shape)
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
        
        
layer1 = Layer_Dense(X.shape[1], 5)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(5, 10)
layer3 = Layer_Dense(10, 5)
layer4 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.outputs)
activation1.forward(layer1.outputs)
print(activation1.outputs)
# layer2.forward(activation1.outputs)
# layer3.forward(layer2.outputs)
# layer4.forward(layer3.outputs)
# print(layer4.outputs)
