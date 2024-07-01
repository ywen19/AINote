"""
Inspired by: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
Different decay methods for learning rate. 
"""

import numpy as np

class StepDecay(object):
    """
    A typical way of decay to drop the learning rate by half every 10 epochs.
    Formular is: lr = lr0 * drop^floor(epoch / epochs_drop) 
    """
    
    def __init__(self, init_lr, drop_rate=0.5, drop_frequence=10):
        self.init_lr = init_lr
        self.drop_rate = drop_rate
        self.drop_frequence = drop_frequence
        
    def decay(self, epoch):
        # epoch is the current epoch/step 
        return self.init_lr * np.power(self.drop_rate, np.floor(epoch/self.drop_frequence))
    
    
    
class ExponentialDecay(object):
    """
    A common schedule.
    Formular is: lr = lr0 * e^(−kt)
    Where k is a hyperparameter and t is iteration number
    """
    def __init__(self, init_lr, k=0.1):
        self.init_lr = init_lr
        self.k = k
        
    def decay(self, t):
        # t is iteration number
        return self.init_lr * np.exp(-self.k*t)