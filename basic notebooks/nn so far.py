
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
        
        
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def forward(self, ):
        pass
    

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
            
        
        
        
        
           
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
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.outputs, y)

print("Loss: ", loss)













# print(activation1.outputs)
# print(activation2.outputs[:5])#, '\n', np.sum(activation2.outputs, axis=1, keepdims=True))
# layer2.forward(activation1.outputs)
# layer3.forward(layer2.outputs)
# layer4.forward(layer3.outputs)
# print(layer4.outputs)
