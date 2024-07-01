""" SGD optimizer """

import numpy as np


class SGD():
    def __init__(self, learning_rate, weight_decay):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                
                # use grad_W and grad_b in layer to calculate diff_W and diff_b
                # use weight decay
                
                # W(i+1) = (1-learningrate*decay)*W(i) - learningrate*grad_W
                # ref: https://discuss.pytorch.org/t/how-does-sgd-weight-decay-work/33105/2
                # ref: https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8
                
                layer.W = (1-self.learning_rate*self.weight_decay)*layer.W - self.learning_rate*layer.grad_W
                layer.b -= self.learning_rate*layer.grad_b