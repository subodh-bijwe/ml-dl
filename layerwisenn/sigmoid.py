import numpy as np
from activation_layer import ActivationLayer

class Sigmoid(ActivationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)
    
    
    