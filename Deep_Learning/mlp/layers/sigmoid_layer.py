import numpy as np

class SigmoidLayer():
    def __init__(self):
        self.trainable = False
    
    def forward(self, Input):
        self.cache = 1/(1+np.exp(-Input))
        return self.cache
    
    def backward(self, delta):
        # ref: https://blog.csdn.net/weixin_44478378/article/details/100861801
        # ref: https://blog.csdn.net/github_37462634/article/details/100178257
        return delta*(1-self.cache)*self.cache