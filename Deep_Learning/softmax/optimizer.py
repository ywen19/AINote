import numpy as np
from learnrate import StepDecay, ExponentialDecay

class SGD(object):
    def __init__(self, model, learning_rate, decay_type="momentum", momentum=0.0, drop_rate=0.5, drop_frequence=10, k=0.1):
        self.model = model
        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.momentum = momentum
        
        if self.decay_type == "step":
            self.decay_model = StepDecay(learning_rate, drop_rate, drop_frequence)
        if self.decay_type == "exponential":
            self.decay_model = ExponentialDecay(learning_rate, k)

    def step(self, epoch=1):
        """One updating step, update weights"""

        layer = self.model
        if layer.trainable:
            
            if self.decay_type == "momentum":
                layer.delta_W = self.momentum*layer.delta_W + self.learning_rate*layer.grad_W
                layer.delta_b = self.momentum*layer.delta_b + self.learning_rate*layer.grad_b
            else:
                self.learning_rate = self.decay_model.decay(epoch)
                layer.delta_W = self.learning_rate*layer.grad_W
                layer.delta_b = self.learning_rate*layer.grad_b
            
            layer.W += layer.delta_W
            layer.b += layer.delta_b
        return self.learning_rate
            

