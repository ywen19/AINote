import numpy as np

class ReLULayer():
    def __init__(self):
        self.trainable = False
        
    def forward(self, Input):
        self.cache = np.maximum(0, Input)
        return self.cache
    
    def backward(self, delta):
        delta[self.cache<=0] = 0
        return delta