from layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation, self.activation_prime = activation, activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))