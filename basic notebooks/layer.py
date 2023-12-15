import numpy as np
np.random.seed(0)
X = np.array([
        [1, 2, 3, 2.5], 
        [2.0, 5.0, -1.0, 2.0], 
        [-1.5, 2.7, 3.3, -0.8]
    ])


class Layer_Dense:
    def __init__(self, n_inputs, n_features) -> None:
        # weights are n_features x n_inputs, but to save a transpose operation we do n_inputs x n_features
        self.weights = 0.1 * np.random.randn(n_inputs, n_features)
        # print("weights", self.weights.shape)
        self.biases = np.zeros((1, n_features))
    
    def forward(self, inputs):
        # print("ip and weights", inputs.shape, self.weights.shape)
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 10)
layer3 = Layer_Dense(10, 5)
layer4 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.outputs)
layer3.forward(layer2.outputs)
layer4.forward(layer3.outputs)
print(layer4.outputs)
    

















































"""weights = [
        [0.2, 0.6, 0.9, 1.0], 
        [-0.4, 0.7, 0.45, -0.33], 
        [0.44, -0.65, 0.89, 0.23]
    ]

biases = [1,2,0.4]

weights2 = [
        [0.2, 0.6, 0.9], 
        [-0.4, 0.7, 0.45], 
        [0.44, -0.65, 0.89]
    ]

biases2 = [1,2,0.4]


weights = np.array(weights.T
inputs = np.array(inputs)
biases = np.array(biases.T
weights2 = np.array(weights2.T
biases2 = np.array(biases2.T
layer_1op = np.dot(inputs, weights) + biases # 3 x 3 matrix
layer_2op = np.dot(layer_1op, weights2) + biases2

print(layer_2op)"""



# layer_outputs = []

# for n_weights, n_bias in zip(weights, biases):
#     n_output = 0
#     for n_input, weight in zip(inputs, n_weights):
#         n_output += n_input * weight
#     n_output += n_bias
#     layer_outputs.append(n_output)
    
# print(layer_outputs)

# dot product

# import numpy as np

# input = [1,2,3,4]
# weight = [0.2, 0.6, 0.9, 1.0]
# bias = 2

# op = np.dot(input, weight) + bias

# print(op)