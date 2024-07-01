import numpy as np

class FCLayer():
    def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
        self.num_input = num_input
        self.num_output = num_output
        self.actFunction = actFunction
        self.trainable = trainable
        
        assert self.actFunction in ['relu', 'sigmoid']
        
        self.XavierInit()
        
        self.grad_W = np.zeros((self.num_input, self.num_output))
        self.grad_b = np.zeros((1, self.num_output))
        
        
    def forward(self, Input):
        self.Input = Input
        return np.dot(self.Input, self.W)+self.b
    
    def backward(self, delta):
        self.grad_W = self.Input.T.dot(delta)/self.Input.shape[0]
        self.grad_b = np.mean(delta, axis=0)
        return np.dot(delta, self.W.T)
        
    def XavierInit(self):
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        if self.actFunction == 'relu':
            init_std = raw_std * (2**0.5)
        elif self.actFunction == 'sigmoid':
            init_std = raw_std
        else:
            init_std = raw_std
            
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))